import os
import time


def convert(svgpath="", cover=False, inkpath="C:\Program Files (x86)\Inkscape"):
    """
    CaiShu 2020.04
    批量装换SVG为WMF，以便插入WORD的脚本
    需要提前安装Inkscape软件
    :param svgpath: 包含SVG文件的文件夹地址，应为绝对路径。当在此文件夹打开CMD时，可不输入
    :param cover: 当文件夹有部分SVG转为WMF时，选择不覆盖（cover=False)可以节省时间
    :param inkpath: Inkscape软件安装根目录，修改本py以便重复使用
    :return: None
    """
    os.environ['Path'] += inkpath  # 需要添加到临时变量中
    if svgpath == "":
        svgpath = os.getcwd()
    svglist = [i for i in os.listdir(svgpath) if i.endswith(".svg")]

    if cover == False:
        passlist = [i.replace(".wmf", ".svg") for i in os.listdir(svgpath) if i.endswith(".wmf")]
        svglist = [i for i in svglist if i not in passlist]
    else:
        pass

    if svglist == []:
        print(f"文件夹位置有误，无需要转换的SVG文件!    :   {svgpath}")
    else:
        t1 = time.time()
        for i in range(len(svglist)):
            svg = svgpath + f'\\{svglist[i]}'
            wmf = svgpath + f"\\{svglist[i].replace('.svg', '.wmf')}"
            print(f"{i + 1}/{len(svglist)}   ：   {svg}")
            os.system(f"inkscape -z {svg} -M {wmf}\n")
        t2 = time.time()
        print(f"转换{i + 1}个svg文件完成！ 耗时{round(t2 - t1, 3)} s")


if __name__ == "__main__":
    dirpath = os.getcwd()
    path = dirpath
    path = r"F:\torch_fuzz\fuzz\save\【TE】 CG-SAE"
    inkpath = r"C:\Program Files (x86)\Inkscape"
    inkpath = r"C:\Program Files\Inkscape\bin\inkscape.exe"
    convert(path, cover=True, inkpath=inkpath)