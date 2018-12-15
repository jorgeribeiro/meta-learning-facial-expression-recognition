# Progresso
# Meta-aprendizado aplicado ao problema de reconhecimento de expressões faciais

# ===> Pré-processamento
# Pré-processamento do JAFFE:
	# Detecção de faces via MTCNN (https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
	# Resize para 112 x 150
	# Divisão entre treino (85%) e teste (15%)
	# Feito!

# Pré-processamento do CK+:
	# Algumas imagens não possuem labels, segundo a própria documentação (arquivo ReadMeCohnKanadeDatabase_website.txt), estas serão ignoradas
	# Processo: as imagens serão organizadas da seguinte forma:
		# A primeira imagem de cada sequência serão armazenadas como neutras
		# A partir de (tamanho_sequencia / 2 + 2) serão armazenadas de acordo com a emoção indicada
		# Entre a primeira e (tamanho_sequencia / 2 + 2), as "emoções de transição" não serão incluídas
	# Detecção de faces via MTCNN (https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
	# Resize para 112 x 150
	# Conversão para grayscale
	# Divisão entre treino (85%) e teste (15%)
	# Feito!

# Pré-processamento do FERPlus:
	# Utilizando https://github.com/Microsoft/FERPlus como referência
	# Base aprimorada do FER2013 da Microsoft (FERPlus)
	# Já está dividido entre treino e teste
	# Labels anotados de acordo com porcentagem de emoção:
		# Uma imagem pode ser 80% anger, 10% medo e 10% desconhecido, por exemplo
		# A emoção utilizada será a de maior porcentagem
		# Em casos de empate na maior porcentagem, a imagem correspondente não será utilizada
		# Algumas emoções estão marcadas como 'unknown' ou 'not-a-face', estas também não serão utilizadas
	# Três tamanhos de dataset: full (22k), medium (11k) e small (5k)
	# Detecção de faces via MTCNN (https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
	# Resize para 34 x 44
	# Feito!

# ===> Seleção de arquiteturas
# SimpleNet
	# Treinamentos JAFFE:
		# (I) 100 epochs e batch size 50
		# (II) 300 epochs e batch size 20 
		# (III) 500 epochs e batch size 20: Acurácia: 100%, Teste: ~87%
		# (IV) 1000 epochs, batch size 20 e data aug: Acurácia 100%, Teste: ~90%
		# (V) 500 Epochs, batch size 20: Acurácia 100%, Teste ~88%
		# (VI) (Com otimização) 500 epochs, 20 max_evals (000001): Acurácia: ~99%, Teste: ~95%		

	# Treinamentos CK+: Acurácia: 100%, Teste: ~97% (I)
		# (I) 100 epochs e batch size 20

	# Treinamentos FERPlus: Acurácia: 96%, Teste: ~81% (III)
		# (I) 100 epochs e batch size 20 [SMALL]
		# (II) 100 epochs e batch size 20 [MEDIUM]
		# (III) 100 epochs e batch size 20 [FULL]

# VGG-16
	# Treinamentos JAFFE: Acurácia: ~87%, Teste: ~65% (II)
		# (I) 100 epochs e batch size 20
		# (II) 300 epochs e batch size 20

	# Treinamentos CK+:
		# (I) 100 epochs e batch size 20
		# (II) (Com otimizaçao) 100 epochs, 10 max_evals (000002): Acurácia: 99%, Teste: 99%
		# (II) (Com otimizaçao) 100 epochs, 10 max_evals (000003): Acurácia: 100%, Teste: 97%

	# Treinamentos FERPlus:
		# (I) 100 epochs e batch size 20 [SMALL]
		# (II) 100 epochs e batch size 20 [MEDIUM]
		# (III) 100 epochs e batch size 20 [FULL]
		# (IV) (Com otimização) 100 epochs, 10 max_evals [FULL] (000004): Acurácia: 87%, Teste: 81%
		# (IV) (Com otimização) 100 epochs, 10 max_evals [FULL] (000005): Acurácia: 99%, Teste: 79%

# VGG-19
	# Treinamentos FERPlus: Acurácia: 99%, Teste: 77% (I)
		# (I) 100 epochs e batch size 20 [FULL]

# ResNet-18
	# Treinamentos JAFFE: Acurácia: 98%, Teste: ~53% (I)
		# (I) 100 epochs e batch size 20

	# Treinamentos CK+: Acurácia: 99%, Teste: 87% (I)
		# (I) 100 epochs e batch size 20

	# Treinamentos FERPlus: Acurácia: 96%, Teste: 72% (I)
		# (I) 100 epochs e batch size 20 [FULL]

# ResNet-34
	# Treinamentos FERPlus: Acurácia: 96%, Teste: 75% (I)
		# (I) 100 epochs e batch size 20 [FULL]

# InceptionV3
	# Treinamentos CK+
		# (I) 100 epochs e batch size 20: Acurácia: 100%, Teste: 98%