from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool
from bokeh.io import output_notebook

def ncodencorrplot(x, 
                   y1,
                   y2,
                   y3,
                   y_label,
                   title,
                   filename):

    # 创建绘图对象
    # p = figure(y_axis_type="log", x_axis_label='X', y_axis_label='Y', title='Plot with Log Scale')
    p = figure(y_axis_type="log", title=title)

    # 添加曲线
    p.line(x, y1, legend_label='ZF', line_color='blue')
    p.line(x, y2, legend_label='LMMSE', line_color='green')
    p.line(x, y3, legend_label='DIP', line_color='red')

    # 添加鼠标悬停工具
    hover_tool = HoverTool(tooltips=[("x", "@x"), ("y", "@y")])
    p.add_tools(hover_tool)

    # 设置标题
    # p.title.text = title
    p.title.align = "center" # type: ignore
    p.title.text_font_style = "bold" # type: ignore
    p.title.text_font_size = "12pt" # type: ignore

    # 设置横轴标签
    p.xaxis.axis_label = "Eb/N0 (dB)"
    p.xaxis.axis_label_text_font_style = "bold"
    p.xaxis.axis_label_text_font_size = "10pt"

    # 设置纵轴标签
    p.yaxis.axis_label = y_label
    p.yaxis.axis_label_text_font_style = "bold"
    p.yaxis.axis_label_text_font_size = "10pt"

    # 设置图例字体样式和大小
    p.legend.label_text_font_style = "normal"
    p.legend.label_text_font_size = "8pt"

    # 设置图例位置为右上角
    p.legend.location = 'bottom_left'

    # 在编译器中显示图形
    output_notebook()

    # 在默认浏览器中打开 HTML 文件
    # import webbrowser
    # webbrowser.open("plot.html")

    # 设置输出文件
    # output_file("plot.html")
    output_file(filename)
    
    # 显示图形
    # show(p, browser=None)
    show(p)

    return p