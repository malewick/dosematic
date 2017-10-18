import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def Someotherstuff():

    fig = plt.figure(figsize=(6,4))
    plt.axis('off')

    data = [[  66386,  174296,   75131,  577908,   32015],
	    [  58230,  381139,   78045,   99308,  160454],
	    [  89135,   80552,  152558,  497981,  603535],
	    [  78415,   81858,  150656,  193263,   69638],
	    [ 139361,  331509,  343164,  781380,   52269]]

    columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
    rows = ['1 ','2','3','4','5']

    cell_text = []
    for row in data:
	cell_text_row=[]
	for cell in row:
	    cell_text_row.append(str(cell))
	cell_text.append(cell_text_row)

    the_table = plt.table(cellText=cell_text,
	    rowLabels=rows,
	    colLabels=columns,
	    loc='center',
	    bbox=[0.15, 0.15, 0.85, 0.85])
    the_table.set_fontsize(10)
    the_table.scale(1, 1.5)

    #plt.subplots_adjust(left=0.15, top=0.9)

    return fig

def TitleSlide(text):
    fig=plt.figure(figsize=(6,4))
    plt.text(0.01,0.90,text,fontsize=15)
    plt.axis('off')
    return fig
 
pdf=PdfPages('report.pdf')

fig=TitleSlide(text="""
This is some sample unformatted text.
I'd like to put this into a table of some sort.
""")

pdf.savefig(fig)

pdf.savefig(Someotherstuff())

pdf.close()
