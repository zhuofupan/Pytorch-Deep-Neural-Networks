import matplotlib.pyplot as plt
import subprocess
import os
inkscapePath = r"C:\Program Files\Inkscape\bin\inkscape.exe"

def exportEmf(savePath, plotName, fig=None, keepSVG=False):
    """Save a figure as an emf file

    Parameters
    ----------
    savePath : str, the path to the directory you want the image saved in
    plotName : str, the name of the image 
    fig : matplotlib figure, (optional, default uses gca)
    keepSVG : bool, whether to keep the interim svg file
    """

    figFolder = savePath + r"/{}.{}"
    svgFile = os.path.abspath(figFolder.format(plotName,"svg"))
    emfFile = os.path.abspath(figFolder.format(plotName,"emf"))
    if fig:
        use=fig
    else:
        use=plt
    use.savefig(svgFile, bbox_inches='tight')
    
    print('Save emf in {}'.format(emfFile))
    os.environ['Path'] += r"C:\Program Files\Inkscape"
    # subprocess.run(['inkscape', svgFile, '--export-pdf={}'.format(emfFile), '--without-gui'])
    os.system(f"inkscape -z svgFile -M emfFile\n")
    
    # if not keepSVG:
    #     os.system('del "{}"'.format(svgFile))
        
if __name__ == '__main__':
    import numpy as np
    tt = np.linspace(0, 2*3.14159)
    plt.plot(tt, np.sin(tt))
    exportEmf(r"C:\Users\userName", 'FileName')