# GreenSegmenter — diretrizes de desenvolvimento

Aplicativo local de anotação binária de vegetação. Fluxo: carregar imagens,
recortar/filtrar tiles, gerar máscaras com ExG ou redes neurais, revisar e exportar.

- Inspecione código e testes antes de alterar funcionalidades.
- Preserve arquivos originais, checkpoints, sessões e mudanças existentes.
- Use Python, Gradio Blocks, Pillow, NumPy e OpenCV. PyTorch é opcional.
- Mantenha interface em português do Brasil e nomes de código em inglês.
- Separe UI, processamento, inferência, persistência e exportação.
- Não inclua caminhos absolutos de máquinas, credenciais, imagens pessoais,
  checkpoints ou resultados experimentais na distribuição.
- Dados gerados ficam em workspace/ e são ignorados pelo Git.
- CPU é o padrão. Não apresente inferências simuladas como resultados reais.
- Preserve coordenadas e dimensões válidas dos tiles.
- PNG binário é a fonte de verdade; COCO é uma representação derivada.
- Edição manual acontece no navegador. Evite callbacks a cada pincelada e
  atualizações recursivas do ImageEditor.
- Use apply_patch para edições e preserve mudanças não relacionadas.
- Teste máscaras, transformações, exportação e persistência com dados pequenos.
- Testes neurais devem usar modelos controlados, sem checkpoints grandes no Git.
- Execute verificações relevantes e informe limitações de validação.
- Avaliação, geometria, treinamento, autenticação e serviços remotos estão
  fora do escopo.
