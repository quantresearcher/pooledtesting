"""Contains parameters for the Bokeh app."""
import os


BASE_DIR = os.path.dirname(__file__)

APP_TITLE = 'Pooled Testing'
NUM_TRIALS = 20000
RANDOM_STATE = 42

########################### Header ###########################
HEADER_FILENAME = os.path.join(BASE_DIR, 'templates', 'header.html')
HEADER_SIZING_MODE = 'stretch_width'

########################### Footer ############################
FOOTER_FILENAME = os.path.join(BASE_DIR, 'templates', 'footer.html')
FOOTER_SIZING_MODE = 'stretch_width'

########################### Figure ############################
FIGURE_TOOLS = 'reset'
FIGURE_TOOLBAR_LOCATION = None
FIGURE_ASPECT_RATIO = 2
FIGURE_SIZING_MODE = 'scale_both'
FIGURE_MAX_WIDTH = 1600
FIGURE_TITLE_ALIGN = 'center'
FIGURE_TITLE_FONT_SIZE = '20pt'
FIGURE_TITLE_FONT_STYLE = 'normal'
FIGURE_X_AXIS_LABEL = 'Pool size'
FIGURE_Y_AXIS_LABEL = 'Expected number of tests per person'
FIGURE_X_AXIS_LABEL_STANDOFF = 10
FIGURE_Y_AXIS_LABEL_STANDOFF = 15
FIGURE_X_AXIS_FONT_SIZE = '16pt'
FIGURE_Y_AXIS_FONT_SIZE = '16pt'
FIGURE_X_AXIS_FONT_STYLE = 'normal'
FIGURE_Y_AXIS_FONT_STYLE = 'normal'
FIGURE_X_AXIS_MANTISSAS = [1, 2, 5]
FIGURE_X_AXIS_MIN_INTERVAL = 1
FIGURE_X_AXIS_DESIRED_NUM_TICKS = 40
LINE_WIDTH = 2
CIRCLE_COLOR = 'red'
CIRCLE_SIZE = 10
HOVER_TOOLTIPS = [
    ('Pool size', '@x'),
    ('Expected number of tests per person', '@y{0.000}'),
]
HOVER_MODE = 'vline'

########################### Controls ############################
PREVALENCE_TITLE = 'Prevalence of COVID-19'
PREVALENCE_START = 0
PREVALENCE_END = 1
PREVALENCE_VALUE = 0.02
PREVALENCE_STEP = 0.01
PREVALENCE_FORMAT = '0%'
PREVALENCE_SIZING_MODE = 'stretch_width'

CORRELATION_TITLE = 'Correlation'
CORRELATION_START = 0
CORRELATION_END = 1
CORRELATION_VALUE = 0
CORRELATION_STEP = 0.01
CORRELATION_FORMAT = '0%'
CORRELATION_SIZING_MODE = 'stretch_width'

SENSITIVITY_TITLE = 'Sensitivity'
SENSITIVITY_START = 0.5
SENSITIVITY_END = 1
SENSITIVITY_VALUE = 0.99
SENSITIVITY_STEP = 0.01
SENSITIVITY_FORMAT = '0%'
SENSITIVITY_SIZING_MODE = 'stretch_width'

SPECIFICITY_TITLE = 'Specificity'
SPECIFICITY_START = 0.5
SPECIFICITY_END = 1
SPECIFICITY_VALUE = 0.99
SPECIFICITY_STEP = 0.01
SPECIFICITY_FORMAT = '0%'
SPECIFICITY_SIZING_MODE = 'stretch_width'

POOL_SIZE_TITLE = 'Pool sizes to consider'
POOL_SIZE_START = 2
POOL_SIZE_END = 100
POOL_SIZE_VALUE = (2, 40)
POOL_SIZE_STEP = 1
POOL_SIZE_SIZING_MODE = 'stretch_width'

############################ Inputs #############################
INPUTS_SIZING_MODE = 'fixed'
INPUTS_WIDTH = 380
INPUTS_HEIGHT = 500

############################ Layout ############################
LAYOUT_SIZING_MODE = 'stretch_both'