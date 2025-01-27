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
    merged_name = 'HaoFu_ApplicationDocuments_II-856_23.pdf'
    if is_file_found:

        merger = PyPDF2.PdfMerger()
        files = ['cover.pdf', 'TU_berlin_Generative_Methods.pdf', 'CV.pdf', 'master.pdf',
                 'HaoFu_Publication_LCE_BoltPosePK_en.pdf', 'Reference_Wang.pdf', 'Reference_Zhang.pdf'
                 ]
        for filename in files:
            merger.append(PyPDF2.PdfReader(os.path.join(dir, filename)))
        merger.write(os.path.join(dir, merged_name))
    else:
        print(os.listdir(dir))


def _pipline_merge_3_pdfs(is_file_found=True):
    import PyPDF2
    import os

    dir = r'C:\Users\Lenovo\Desktop'
    merged_name = 'YingLuo_ApplicationDocuments_V000007189.pdf'
    if is_file_found:

        merger = PyPDF2.PdfMerger()
        files = ['cover.pdf', 'CV.pdf', 'CoverLetter.pdf', 'master.pdf']
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


import os
import subprocess


def pdf_to_svg(pdf_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(pdf_path))[0] + ".svg")
    command = ['pdf2svg', pdf_path, output_dir]
    # command = f"pdf2svg {pdf_path} {output_file} 1"
    subprocess.run(command, check=True)
    print(f"SVG file saved to {output_file}")


def _pdf_to_svg():
    pdf_to_svg(r'D:/master/semester5/HiWi/Publication/Presentation/table1.pdf',
               r'D:/master/semester5/HiWi/Publication/Presentation')


def audio_segment():
    from moviepy.editor import AudioFileClip, concatenate_audioclips, AudioClip

    audio = AudioFileClip(
        "D:/master/semester5/HiWi/Publication/Presentation/voice/1_Introduction_remanufacturing_4.mp3")

    one_second_silence = AudioClip(lambda t: 0, duration=0.5)

    new_audio = concatenate_audioclips([audio, one_second_silence])

    new_audio.write_audiofile(
        "D:/master/semester5/HiWi/Publication/Presentation/voice/1_Introduction_remanufacturing_4_2.mp3")


if __name__ == "__main__":
    audio_segment()
