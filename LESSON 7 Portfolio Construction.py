# Import Algorithm API
import quantopian.algorithm as algo

# Import Optimize API
import quantopian.optimize as opt

# Pipeline imports
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.factors import SimpleMovingAverage

# Import built-in universe and Risk API method
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline


def initialize(context):
    # Constraint parameters
    context.max_leverage = 1.0
    context.max_pos_size = 0.015
    context.max_turnover = 0.95

    # Attach data pipelines
    algo.attach_pipeline(
        make_pipeline(),
        'data_pipe'
    )
    algo.attach_pipeline(
        risk_loading_pipeline(),
        'risk_pipe'
    )

    # Schedule rebalance function
    algo.schedule_function(
        rebalance,
        algo.date_rules.week_start(),
        algo.time_rules.market_open(),
    )


def before_trading_start(context, data):
    # Get pipeline outputs and
    # store them in context
    context.pipeline_data = algo.pipeline_output('data_pipe')

    context.risk_factor_betas = algo.pipeline_output('risk_pipe')


# Pipeline definition
def make_pipeline():

    sentiment_score = SimpleMovingAverage(
        inputs=[stocktwits.bull_minus_bear],
        window_length=3,
        mask=QTradableStocksUS()
    )

    return Pipeline(
        columns={
            'sentiment_score': sentiment_score,
        },
        screen=sentiment_score.notnull()
    )


def rebalance(context, data):
    # Retrieve alpha from pipeline output
    alpha = context.pipeline_data.sentiment_score

    if not alpha.empty:
        # Create MaximizeAlpha objective
        objective = opt.MaximizeAlpha(alpha)

        # Create position size constraint
        constrain_pos_size = opt.PositionConcentration.with_equal_bounds(
            -context.max_pos_size,
            context.max_pos_size
        )

        # Constrain target portfolio's leverage
        max_leverage = opt.MaxGrossExposure(context.max_leverage)

        # Ensure long and short books
        # are roughly the same size
        dollar_neutral = opt.DollarNeutral()

        # Constrain portfolio turnover
        max_turnover = opt.MaxTurnover(context.max_turnover)

        # Constrain target portfolio's risk exposure
        # By default, max sector exposure is set at
        # 0.2, and max style exposure is set at 0.4
        factor_risk_constraints = opt.experimental.RiskModelExposure(
            context.risk_factor_betas,
            version=opt.Newest
        )

        # Rebalance portfolio using objective
        # and list of constraints
        algo.order_optimal_portfolio(
            objective=objective,
            constraints=[
                constrain_pos_size,
                max_leverage,
                dollar_neutral,
                max_turnover,
                factor_risk_constraints,
            ]
        )
