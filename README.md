# Jupyter Notebook interativo para cálculo de parâmetros hemodinâmicos de vídeos de coração do inseto

<font size=5>Autores: [Felipe Hiroshi Kano Inazumi](mailto:f215696@dac.unicamp.br),
[Nelly Catherine Barbosa Calderon](mailto:n160942@dac.unicamp.br), 
[Adriano Santana](mailto:adriano.rsantana@gmail.com),
[Rosana Almada Bassani](mailto:arbassani@unicamp.br), 
[José Wilson Magalhães Bassani](bassani@unicamp.br)

<p style='text-align: justify;'> Este projeto tem como objetivo fazer o cálculo de parâmetros hemodinâmicos como débito cardíaco, fração de ejeção, volume de ejeção e fração de encurtamento a partir de vídeos contendo a atividade contrátil do coração do inseto </p> 

## Execução Online  [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fehiroshii/JupyterNTB/blob/main/online/google_colab/main_ntb.ipynb)

O software também pode ser executado de forma online em um ambiente Google Colab, sem a necessidade de instalação dos arquivos. Para que o vídeo possa ser lido pelo software, é necessário que o vídeo a ser analisado seja enviado na nuvem. Os processos para envio são descritos abaixo


 1. **Envio do Vídeo no Google Drive**

Faça login em seu *[Google Drive](https://drive.google.com/drive/u/0/my-drive)* e faça o envio do vídeo a ser analisado.

2. **Execute o _Software_**

Abra o [link do software](https://colab.research.google.com/github/fehiroshii/JupyterNTB/blob/main/online/google_colab/main_ntb.ipynb)  e execute o programa em "Ambiente de execução" -> "Executar tudo", ou pressione a tecla de atalho Ctrl + F9

3. **Carregue o arquivo**
O seguinte menu será apresentado:

Se o vídeo foi carregado no Google Drive, faça o login clicando no botão "Login Google Drive" e 

Após isso, coloque o nome do vídeo no caminho "my_path" como a seguir: 
  * Se o vídeo está localizado dentro da raíz do Google Drive, será necessário apenas inserir o nome do vídeo, por exemplo: "my_video.avi"

  * Se o vídeo está inserido dentro alguma pasta, será necessário colocar o caminho da pasta antes do nome do vídeo, por exemplo: se o vídeo está dentro da pasta "my_file", coloque "/my_file/my_video.avi"

Por fim,  pressione o botão "Load Video" 


###

## Instalação

1. Baixe os arquivos da pasta /source

   Alternativamente, faça o dowload do projeto executando o seguinte comando

   ```sh
   $git clone https://github.com/fehiroshii/JupyterNTB/
   ```
2. Execute o arquivo "main_ntb.ipynb"




### Pré-requisitos

Para que o software execute corretamente, são necessários a instalação da linguagem Python, de um ambiente Jupyter e das bibliotecas explicitadas abaixo:

- Python  
- Jupyter Notebook ou JupyterLab


| Biblioteca                             |  Versão   |
| :------------------------------------: | :-------: |
| [Matplotlib](https://matplotlib.org/)  | 3.5.1     |
| [Numpy](https://numpy.org/)            |  1.20.1   |
| [OpenCV](https://opencv.org/)          |  4.6.0.   |
| [Scipy](https://scipy.org/)            |  1.8.1    |
| [Pandas](https://pandas.pydata.org/)   | 1.2.3     |

  

## Referências

1.	Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T, Cournapeau D, et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nat Methods. 2020;17:261–72. 
2.	Harris CR, Millman KJ, der Walt SJ van, Gommers R, Virtanen P, David Cournapeau, et al. Array programming with NumPy. Nature [Internet]. 2020 set;585(7825):357–62. Available from: https://doi.org/10.1038/s41586-020-2649-2
3.	pandas development team T. pandas-dev/pandas: Pandas [Internet]. Zenodo; 2020. Available from: https://doi.org/10.5281/zenodo.3509134
4.	Bradski G. The OpenCV Library. Dr Dobb’s Journal of Software Tools. 2000; 
5.	Hunter JD. Matplotlib: A 2D graphics environment. Comput Sci Eng. 2007;9(3):90–5. 