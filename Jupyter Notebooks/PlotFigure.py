from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool
from bokeh.io import output_notebook

def PlotFigure(self, 
        x, 
        y, 
        legend, 
        line_color,
        x_label, 
        y_label, 
        title):

    # 生成示例数据
    # x = [1, 2, 3, 4, 5]
    # y1 = [1, 10, 100, 1000, 10000]
    # y2 = [2, 20, 200, 2000, 20000]
    # y3 = [3, 30, 300, 3000, 30000]

    # 创建绘图对象
    # p = figure(y_axis_type="log", x_axis_label='X', y_axis_label='Y', title='Plot with Log Scale')
    p = figure(y_axis_type="log")

    # 添加曲线
    # p.line(x, y1, legend_label='Curve 1', line_color='blue')
    # p.line(x, y2, legend_label='Curve 2', line_color='red')
    # p.line(x, y3, legend_label='Curve 3', line_color='green')
    p.line(x, y, legend_label=legend, line_color=line_color)

    # 添加鼠标悬停工具
    hover_tool = HoverTool(tooltips=[("x", "@x"), ("y", "@y")])
    p.add_tools(hover_tool)

    # 设置标题
    p.title.text = title # type: ignore
    p.title.align = "center" # type: ignore
    p.title.text_font_style = "italic" # type: ignore
    p.title.text_font_size = "18pt" # type: ignore

    # 设置横轴标签
    p.xaxis.axis_label = x_label
    p.xaxis.axis_label_text_font_style = "bold"
    p.xaxis.axis_label_text_font_size = "12pt"

    # 设置纵轴标签
    p.yaxis.axis_label = y_label
    p.yaxis.axis_label_text_font_style = "bold"
    p.yaxis.axis_label_text_font_size = "12pt"

    # 设置图例字体样式和大小
    p.legend.label_text_font_style = "italic"
    p.legend.label_text_font_size = "10pt"

    # 设置图例位置为右上角
    p.legend.location = 'bottom_right'

    # 在编译器中显示图形
    output_notebook()

    # # 设置输出文件
    # output_file("plot.html")

    # 显示图形
    show(p)
    # show(p, browser=None)

    # # 在默认浏览器中打开 HTML 文件
    # import webbrowser
    # webbrowser.open("plot.html")