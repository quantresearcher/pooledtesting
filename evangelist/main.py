"""A Bokeh app with a description at the top, controls in a panel on the left,
a figure on the right, and more text at the bottom.
"""
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, layout
from bokeh.models import (
    ColumnDataSource,
    Div,
    HoverTool,
    NumeralTickFormatter,
    RangeSlider,
    Slider
)
from bokeh.plotting import figure
from .parameters import *
from .utils import (
    get_number_of_tests_exact,
    get_number_of_tests_simulated
)


header = Div(text = open(HEADER_FILENAME).read(),
             sizing_mode = HEADER_SIZING_MODE)
footer = Div(text = open(FOOTER_FILENAME).read(),
             sizing_mode = FOOTER_SIZING_MODE)
line_source = ColumnDataSource({'x': [], 'y': []})
circle_source = ColumnDataSource({'optimal_pool_size': [], 'optimal_tests_per_person': []})

############################# Figure ##############################
fig = figure(tools = FIGURE_TOOLS,
             toolbar_location = FIGURE_TOOLBAR_LOCATION,
             aspect_ratio = FIGURE_ASPECT_RATIO,
             max_width = FIGURE_MAX_WIDTH,
             sizing_mode = FIGURE_SIZING_MODE)
line = fig.line(x = 'x',
                y = 'y',
                source = line_source,
                width = LINE_WIDTH)
circle = fig.circle(x = 'optimal_pool_size',
               y = 'optimal_tests_per_person',
               source = circle_source,
               color = CIRCLE_COLOR,
               size = CIRCLE_SIZE)
fig.title.align = FIGURE_TITLE_ALIGN
fig.title.text_font_size = FIGURE_TITLE_FONT_SIZE
fig.title.text_font_style = FIGURE_TITLE_FONT_STYLE
fig.xaxis.axis_label = FIGURE_X_AXIS_LABEL
fig.yaxis.axis_label = FIGURE_Y_AXIS_LABEL
fig.xaxis.axis_label_standoff = FIGURE_X_AXIS_LABEL_STANDOFF
fig.yaxis.axis_label_standoff = FIGURE_Y_AXIS_LABEL_STANDOFF
fig.xaxis.axis_label_text_font_size = FIGURE_X_AXIS_FONT_SIZE
fig.yaxis.axis_label_text_font_size = FIGURE_Y_AXIS_FONT_SIZE
fig.xaxis.axis_label_text_font_style = FIGURE_X_AXIS_FONT_STYLE
fig.yaxis.axis_label_text_font_style = FIGURE_Y_AXIS_FONT_STYLE
fig.yaxis.minor_tick_line_color = None
fig.xaxis.minor_tick_line_color = None
fig.xaxis.ticker.mantissas = FIGURE_X_AXIS_MANTISSAS
fig.xaxis.ticker.min_interval = FIGURE_X_AXIS_MIN_INTERVAL
fig.xaxis.ticker.desired_num_ticks = FIGURE_X_AXIS_DESIRED_NUM_TICKS
hover = HoverTool()
hover.tooltips = HOVER_TOOLTIPS
hover.mode = HOVER_MODE
hover.renderers = [line]
fig.add_tools(hover)

############################# Controls ##############################
prevalence_control = Slider(title = PREVALENCE_TITLE,
                            start = PREVALENCE_START,
                            end = PREVALENCE_END,
                            value = PREVALENCE_VALUE,
                            step = PREVALENCE_STEP,
                            format = NumeralTickFormatter(format = PREVALENCE_FORMAT),
                            sizing_mode = PREVALENCE_SIZING_MODE)
correlation_control = Slider(title = CORRELATION_TITLE,
                             start = CORRELATION_START,
                             end = CORRELATION_END,
                             value = CORRELATION_VALUE,
                             step = CORRELATION_STEP,
                             format = NumeralTickFormatter(format = CORRELATION_FORMAT),
                             sizing_mode = CORRELATION_SIZING_MODE)
sensitivity_control = Slider(title = SENSITIVITY_TITLE,
                             start = SENSITIVITY_START,
                             end = SENSITIVITY_END,
                             value = SENSITIVITY_VALUE,
                             step = SENSITIVITY_STEP,
                             format = NumeralTickFormatter(format = SENSITIVITY_FORMAT),
                             sizing_mode = SENSITIVITY_SIZING_MODE)
specificity_control = Slider(title = SPECIFICITY_TITLE,
                             start = SPECIFICITY_START,
                             end = SPECIFICITY_END,
                             value = SPECIFICITY_VALUE,
                             step = SPECIFICITY_STEP,
                             format = NumeralTickFormatter(format = SPECIFICITY_FORMAT),
                             sizing_mode = SPECIFICITY_SIZING_MODE)
pool_size_control = RangeSlider(title = POOL_SIZE_TITLE,
                                start = POOL_SIZE_START,
                                end = POOL_SIZE_END,
                                value = POOL_SIZE_VALUE,
                                step = POOL_SIZE_STEP,
                                sizing_mode = POOL_SIZE_SIZING_MODE)
controls = [prevalence_control,
            correlation_control,
            sensitivity_control,
            specificity_control,
            pool_size_control]
inputs = column(*controls,
                width = INPUTS_WIDTH,
                height = INPUTS_HEIGHT,
                sizing_mode = INPUTS_SIZING_MODE)


def update():
    pool_sizes = np.arange(start = int(pool_size_control.value[0]),
                           stop = int(pool_size_control.value[1]) + 1)
    if correlation_control.value == 0: # separating out this case for efficiency
        tests_per_person = get_number_of_tests_exact(prevalence = prevalence_control.value,
                                                     sensitivity = sensitivity_control.value,
                                                     specificity = specificity_control.value,
                                                     pool_sizes = pool_sizes)
    else:
        tests_per_person = get_number_of_tests_simulated(prevalence = prevalence_control.value,
                                                         correlation = correlation_control.value,
                                                         sensitivity = sensitivity_control.value,
                                                         specificity = specificity_control.value,
                                                         pool_sizes = pool_sizes,
                                                         num_trials = NUM_TRIALS,
                                                         random_state = RANDOM_STATE)
    tests_per_person = pd.Series(tests_per_person)
    line_source.data = {
        'x': tests_per_person.index,
        'y': tests_per_person
    }
    optimal_pool_size = tests_per_person.idxmin()
    optimal_tests_per_person = tests_per_person[optimal_pool_size]
    circle_source.data = {
        'optimal_pool_size': [optimal_pool_size],
        'optimal_tests_per_person': [optimal_tests_per_person]
    }
    fig.title.text = f'Optimal pool size: {optimal_pool_size:.0f}      ' \
                     f'Reduction in tests: {round(100 * (1 - optimal_tests_per_person), 1)}%'


def precompute():
    if correlation_control.value != 0:
        pool_sizes = np.arange(start = POOL_SIZE_START,
                               stop = POOL_SIZE_END + 1)
        get_number_of_tests_simulated(prevalence = prevalence_control.value,
                                      correlation = correlation_control.value,
                                      sensitivity = sensitivity_control.value,
                                      specificity = specificity_control.value,
                                      pool_sizes = pool_sizes,
                                      num_trials = NUM_TRIALS,
                                      random_state = RANDOM_STATE)
    

for control in controls:
    control.on_change('value', lambda attr, old, new: update())
for control in controls[:-1]:
    control.on_change('value_throttled', lambda attr, old, new: precompute())

######################### Document Structure ##########################
curdoc().title = APP_TITLE
curdoc().add_root(
    layout(
        [
            [header],
            [inputs, fig],
            [footer]
        ],
        sizing_mode = LAYOUT_SIZING_MODE
    )
)
update() # load initial data into figure