def _pipline_merge_pdf(is_file_found=False):
    import PyPDF2
    import os

    dir = r'C:\Users\Lenovo\OneDrive\MV\print'
    merged_name = 'MV_merged.pdf'
    if is_file_found:

        merger = PyPDF2.PdfMerger()
        files = ['01-Introduction.pdf',
                 '02-Preprocessing.pdf',
                 '03-Edgedetection.pdf',
                 '04-Curvefitting.pdf',
                 '05-Color.pdf',
                 '06-Segmentation.pdf',
                 '07-Optics.pdf',
                 '10-Patternrecognition.pdf',
                 '12-DeepLearning.pdf']
        for filename in files:
            merger.append(PyPDF2.PdfReader(os.path.join(dir, filename)))
        merger.write(os.path.join(dir, merged_name))
    else:
        print(os.listdir(dir))


def _pipline_merge_2_pdfs(is_file_found=True):
    import PyPDF2
    import os


    dir = r'C:\Users\Lenovo\Desktop'
    merged_name = 'HaoFu_ApplicationDocuments_V-803_23.pdf'
    if is_file_found:

        merger = PyPDF2.PdfMerger()
        files = ['cover.pdf', 'KIT_TU_Berlin_4_0.pdf', 'CV.pdf', 'master.pdf',
                 'HaoFu_Publication_LCE_BoltPosePK_en.pdf','HaoFu_MA_Proposal.pdf'
                 ]
        for filename in files:
            merger.append(PyPDF2.PdfReader(os.path.join(dir, filename)))
        merger.write(os.path.join(dir, merged_name))
    else:
        print(os.listdir(dir))


def png_to_gif(png_dir, gif_dir, duration_time):
    import imageio
    import os

    os.chdir(png_dir)
    file_list = os.listdir()
    frames = []
    for png in file_list:
        if 'frame_000' in png:
            frames.append(imageio.imread(png))
    imageio.mimsave(gif_dir, frames, 'GIF', duration=duration_time)


def _png_to_gif():
    png_to_gif(r'C:/Users/Lenovo/Desktop', 'C:/Users/Lenovo/Desktop/test.gif', 10)


if __name__ == "__main__":
    _pipline_merge_2_pdfs()
