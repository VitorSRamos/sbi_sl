# sbi_sl
Files for strong lensing parameter inference pipeline

-------------------------------------------------------------------------------------------------------------------
Simulação de imagens com deeplenstronomy na pasta img_sim. Imagens são geradas pelo notebook img_sim.ipynb. Os detalhes das simulações são definidos por configuration_file.yaml (raio de einstein e dispersão de velocidades) e por data/user_distribution.txt (magnitudes e redshifts). Simulações são salvas em ExampleDataset, são 3 arquivos: um npy de images, um csv de parametros (chamado metadata), que contém um milhão de colunas, mas são filtradas em data_prep. gera tbm um npy chamado sim_dicts que não é usado.

-------------------------------------------------------------------------------------------------------------------

Preparação dos dados em data_prep/data_prep.ipynb. comentários no código. vai precisar dos arquivos clean_4band_images.npy e clean_lens_params.csv, que foram grandes demais pro github. o processamento gera vários arquivos que são salvos em data_prep/data.

------------------------------------------------------------------------------------------------------------------

A inferencia é rodada no pipeline.py em loop sobre cada parametro e cada arquitetura pra testarusando os arquivos de data_prep/data. Comentários no código. As funções estão nos arquivos em SBI4SL, que está estruturado (mt porcamente) como um pacote. O arquivo pipeline.py gera uma pasta de resultados e varias subpastas para parametros e arquiteturas.

------------------------------------------------------------------------------------------------------------------
