from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool
from bokeh.io import output_notebook

def plotycodeycorr(x, 
                    # y1,
                    # y2,
                    # y3,
                    # y4,
                    # y5,
                    # y6,
                    # y7,
                    # y8,
                    # y9,
                    y10,
                    y11,
                    y12,
                    y_label,
                    title,
                    filename):

    p = figure(y_axis_type="log", title=title)

    # p.line(x, y1, legend_label='ZF SER', line_color='violet')
    # p.line(x, y2, legend_label='LMMSE SER', line_color='turquoise')
    # p.line(x, y3, legend_label='DIP SER', line_color='orange')
    # p.line(x, y4, legend_label='Coded ZF SER', line_color='dodgerblue')
    # p.line(x, y5, legend_label='Coded LMMSE SER', line_color='lime')
    # p.line(x, y6, legend_label='Coded DIP SER', line_color='darkred')
    # p.line(x, y7, legend_label='ZF BER', line_color='purple')
    # p.line(x, y8, legend_label='LMMSE BER', line_color='lightseagreen')
    # p.line(x, y9, legend_label='DIP BER', line_color='gold')
    p.line(x, y10, legend_label='Coded ZF BER', line_color='blue')
    p.line(x, y11, legend_label='Coded LMMSE BER', line_color='green')
    p.line(x, y12, legend_label='Coded DIP BER', line_color='red')

    hover_tool = HoverTool(tooltips=[("x", "@x"), ("y", "@y")])
    p.add_tools(hover_tool)

    # p.title.text = title
    p.title.align = "center" # type: ignore
    p.title.text_font_style = "bold" # type: ignore
    p.title.text_font_size = "12pt" # type: ignore

    p.xaxis.axis_label = "Eb/N0 (dB)"
    p.xaxis.axis_label_text_font_style = "bold"
    p.xaxis.axis_label_text_font_size = "10pt"

    p.yaxis.axis_label = y_label
    p.yaxis.axis_label_text_font_style = "bold"
    p.yaxis.axis_label_text_font_size = "10pt"

    p.legend.label_text_font_style = "normal"
    p.legend.label_text_font_size = "8pt"

    p.legend.location = 'bottom_left'

    output_notebook()

    output_file(filename)
    
    show(p)

    return p