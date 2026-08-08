# Scene Finder

App de desktop para Windows que acha mapas de RPG — nos seus arquivos e na internet — a partir de
uma busca só, em português ou inglês.

Ele foi feito porque procurar cena para a sessão significava abrir cinco abas e vasculhar pastas, e
porque as ferramentas prontas que fazem isso demoravam minutos por busca. Aqui a busca local
responde em menos de um décimo de segundo.

![versão](https://img.shields.io/badge/vers%C3%A3o-1.9.0-4ade80) ![licença](https://img.shields.io/badge/licen%C3%A7a-MIT-blue)

## O que ele faz

**Acervo local** — indexa suas pastas de mapas (Foundry VTT ou qualquer outra) e busca por
significado, não por nome de arquivo. Procurar `taverna a noite` acha `Fey Tavern Night` mesmo com a
consulta em português e o arquivo em inglês. **Vídeos e GIFs entram no mesmo índice**: um mapa
animado (`.webm`, `.mp4`, `.gif`) é indexado pelo frame central e aparece com o selo ▶ — o filtro
"Animados" mostra só eles.

**Áudio** — pastas de música e efeitos sonoros são pesquisáveis na mesma busca, com dois sinais:
o nome do arquivo (multilíngue: `passos na neve` acha `Snow_Foley_Walking_On_Snow`) e o **conteúdo
do som** via CLAP — `chuva` acha um `.wav` de chuva mesmo que o arquivo se chame `track_17`. Os
resultados vêm com player embutido.

**Peças separadas de cenas** — acervos de sprites (Forgotten Adventures) ranqueiam numa seção
própria. Sem isso, medido: 148 mil armários e bolos afogavam qualquer mapa no ranking misto.

Nos resultados locais: **clique** abre o visualizador em tela cheia (setas navegam, vídeos tocam,
faixa de variantes embaixo com rolagem pela roda do mouse), **botão direito** copia o caminho no
formato do Foundry, **shift+clique** abre no app padrão e **shift+botão direito** abre a pasta no
Explorer. Mapas que vêm em
várias versões (Day/Night/Gridless…) ocupam um card só, com um selo que expande **todas** as
variantes direto do índice. O botão **≈** procura mapas visualmente parecidos com aquele.

**Fontes online** — a mesma busca consulta em paralelo o Reddit (r/battlemaps, r/dndmaps,
r/FantasyMaps), o Czepeku (separado em fantasy/scifi × scenes/maps) e criadores que você
configurar. Como esses sites só casam texto em inglês, a consulta é traduzida antes — por um
dicionário embutido de termos de cenário e, quando ele não dá conta da frase, por um tradutor
online com cache. A interface mostra qual termo foi usado de fato. Ainda há botões para abrir Lost
Atlas, JamesRPGArt e Google Imagens.

**Filtros por categoria** — a barra Tudo / Scenes / Maps / **Scemap** / Assets / Sounds ranqueia
só a categoria escolhida (cada pasta configurada pertence a uma). Scene e Maps ganham o filtro
extra **▶ Animado**. O Scemap mostra apenas mapas que existem como cena E como mapa de batalha do
mesmo lugar — criadores como o Czepeku publicam os dois. O botão **Fontes** liga/desliga cada
fonte online, e o **⚙ Config** organiza as pastas em quatro categorias (Scenes, Mapas, Assets,
Áudio) com adicionar/remover/interruptor por pasta.

**Busca ao vivo e favoritos** — os resultados locais aparecem enquanto você digita (as fontes
online só são consultadas no Enter, para não gastar cota com consulta pela metade). A ⭐ em cada
card guarda favoritos — o chip **favoritos** filtra por eles, até sem consulta nenhuma — e a tela
inicial mostra seus favoritos e os últimos caminhos copiados. A tecla **/** foca a busca.

**Player persistente** — dar play num som abre a barra "tocando agora" no topo, com pausar, parar
e loop. A música continua tocando enquanto você busca outra coisa — dá para deixar a ambientação
rolando e procurar o mapa da próxima cena ao mesmo tempo.

**⬇ acervo** — resultado online com arquivo público acessível (Reddit, criadores do kemono) pode
ser baixado direto para uma subpasta `Baixados` do seu acervo: ele entra no índice na hora e a
próxima busca já o encontra como local. Fonte que só expõe miniatura (Czepeku é pago) recusa com o
motivo e abre o post original para você comprar.

**Interruptores** — em **⚙ Config** cada pasta indexada e no **Fontes** cada fonte online tem um
interruptor. Desligar
uma pasta a esconde da busca **sem apagá-la do índice**: dá para manter uma pasta de mapas
indexada e ligá-la só quando precisar, sem reindexar de novo a cada troca.

**Pastas de peças** — acervos como o Forgotten Adventures não são mapas, e sim sprites para montar
mapas: pequenos por natureza, e o filtro padrão (≥1024 px) descarta quase tudo. Uma pasta pode ter
regras próprias no `config.json`:

```json
"folders": [
  "X:\\FoundryVTT\\Data\\Assets\\Scenes",
  { "caminho": "X:\\FoundryVTT\\Data\\Assets\\FA",
    "tudo": true, "ignorar": ["FA_Assets"] }
]
```

| Campo | O que faz |
|---|---|
| `tudo` | indexa **toda** imagem da pasta: sem filtro de tamanho e sem descartar tokens, ícones ou miniaturas |
| `min_side` / `min_kb` | limites próprios, quando você não quer nem o padrão nem tudo |
| `ignorar` | pula subpastas pelo nome |

`ignorar` resolve o caso comum de o acervo trazer o mesmo conteúdo duas vezes em formatos
diferentes — no Forgotten Adventures, `FA_Assets` e `FA_Assets_Webp` são as mesmas imagens, e
indexar as duas só geraria trabalho e resultado repetidos.

Combinando com o interruptor, uma pasta de peças pode ficar desligada no dia a dia e ser ligada só
quando você precisa achar uma lápide ou um barril.

## Instalação

Baixe o `SceneFinder-Setup-x.y.z.exe` em [Releases](../../releases) e execute. Instala por usuário,
sem pedir administrador, e cria atalhos no Menu Iniciar e na Área de Trabalho.

Na primeira abertura ele procura sua pasta do Foundry sozinho. Se não achar, ou se seus mapas
estiverem em outro lugar, use **⚙ Pastas** para apontar os diretórios e clicar em salvar — ele
indexa em seguida.

A indexação usa a GPU quando há uma (DirectML: AMD, NVIDIA ou Intel) e cai para o processador
quando não há. Medido numa RX 9070 XT com Ryzen 7: ~500 imagens por minuto. Ela **grava o progresso
periodicamente**, então dá para fechar o app no meio e continuar depois.

**Acrescentar uma pasta não reindexa o resto.** O índice é incremental: imagem já indexada é
reconhecida pelo caminho e data de modificação e nem é aberta de novo. Num acervo de 22.500
imagens, acrescentar uma pasta pequena leva ~12 segundos, e mandar atualizar sem nenhuma mudança
leva ~4. O botão diz quantas imagens são novas antes de começar.

O app avisa quando há versão nova, baixa e abre o instalador — suas configurações são preservadas.
Quando uma atualização troca o modelo de busca, o índice antigo deixa de servir e ele se reconstrói
sozinho na primeira abertura.

## Como a busca local funciona

A consulta em português é ainda combinada com a tradução dela: mediar os dois embeddings sobe o
MRR de 0,828 para 0,950 no gabarito de nomes (`tools/bench_ensemble.py`).

Um modelo de imagem sozinho não resolve battle maps: vistos de cima, quase todos parecem iguais, os
scores empatam e o primeiro resultado vira sorte. O ranking aqui soma três sinais:

1. **Semântica da imagem** — SigLIP2 sobre o conteúdo visual.
2. **Semântica do nome do arquivo** — embedado no mesmo espaço vetorial pelo encoder de texto, que é
   multilíngue. É isso que faz `taverna` alcançar `tavern`, sem tradutor no meio.
3. **Coincidência literal** de palavras do caminho, que vale mais que "parecido".

A escolha do modelo foi medida, não adotada por reputação: `tools/bench_modelos.py` monta consultas
a partir dos nomes dos seus próprios arquivos e mede acerto usando só o sinal visual. Trocar CLIP
ViT-B/32 por SigLIP2 levou o acerto no primeiro resultado de 15% para 25%.

Os dois encoders rodam em ONNX **sem quantização**: em int8 o de imagem cai para 0,72 de fidelidade
e fica pior que o modelo antigo, e o de texto perde 14% de precisão de ranking mesmo parecendo
inofensivo (0,95 de cosseno). Nada vai para a nuvem: nenhuma imagem sua sai do computador, só o
texto da busca chega às fontes online.

### Uma armadilha do DirectML

Toda inferência passa por um lock global no `encoder.py`. Não é excesso de cuidado: o DirectML **não
é thread-safe** e duas chamadas simultâneas derrubam o processo com access violation — às vezes
levando o driver de vídeo junto. Isso acontece pelo caminho mais banal de uso, que é pesquisar
enquanto a indexação roda. `tools/teste_concorrencia.py` reproduz o cenário; sem o lock ele mata o
interpretador em segundos.

## Rodando a partir do código

```bash
git clone https://github.com/RaymundoJMSN/scene-finder
cd scene-finder
powershell -ExecutionPolicy Bypass -File install.ps1   # venv + dependências
venv\Scripts\python tools\export_onnx.py               # gera models/ (~225 MB)
venv\Scripts\python app.py                             # abre a janela
```

`python server.py` sobe só o servidor em `127.0.0.1:8060`, útil para depurar no navegador.

Verificações rápidas:

```bash
venv-build\Scripts\python indexer.py --check           # pipeline de indexação
venv-build\Scripts\python ptbr.py                      # tradução das consultas
venv-build\Scripts\python tools\smoke_encoder.py       # encoder carrega e é multilíngue
venv-build\Scripts\python tools\teste_concorrencia.py  # buscar durante indexar não trava
venv\Scripts\python tools\verify_onnx.py               # ONNX bate com o modelo original
venv\Scripts\python tools\bench_modelos.py             # compara modelos no seu acervo
venv-build\Scripts\python tools\bench_indexacao.py     # velocidade e fidelidade da indexação
venv-build\Scripts\python tools\perfil.py              # onde a indexação gasta tempo
```

Antes de trocar de modelo ou mexer em quantização, rode `bench_modelos.py`: uma escolha ruim aqui
não quebra nada visivelmente, só piora a busca aos poucos.

Para gerar o instalador: `powershell -File build.ps1` (precisa do
[Inno Setup](https://jrsoftware.org/isdl.php)).

## Estrutura

| Arquivo | Responsabilidade |
|---|---|
| `app.py` | Janela nativa (pywebview/WebView2) + servidor embutido |
| `server.py` | Rotas HTTP locais, fontes online, proxy de miniaturas |
| `indexer.py` | Varredura, embeddings, miniaturas, índice incremental |
| `encoder.py` | Inferência SigLIP2 em ONNX |
| `ptbr.py` | Tradução das consultas para as fontes online |
| `updater.py` | Verificação e download de novas versões |
| `index.html` | Interface inteira |

Índice, miniaturas e configuração ficam em `%LOCALAPPDATA%\SceneFinder` e sobrevivem às
atualizações.

## Pastas de áudio

```json
"audio_folders": [
  "X:\\FoundryVTT\\Data\\Assets\\Musicas",
  "X:\\FoundryVTT\\Data\\Assets\\Audio"
]
```

A indexação dos nomes é rápida (minutos). A análise de conteúdo (CLAP) roda depois, uma vez por
arquivo (~4 áudios/s), e é o que permite achar som por descrição. `.m4a` fica só com busca por
nome — o decodificador não lê esse formato.

## Configurando fontes online

As fontes ficam em `config.json` (criado no primeiro uso). Para acompanhar um criador, adicione o
serviço e o ID que aparecem na URL do perfil:

```json
"kemono": [
  { "service": "patreon", "id": "12345678", "name": "Nome do criador" }
]
```

O app lista títulos e miniaturas e abre o post original no navegador; o botão ⬇ só salva, para uso
próprio, arquivos que a fonte já expõe publicamente. Nada é redistribuído. Compre nos criadores que
você usa.

## Licença

MIT — veja [LICENSE](LICENSE). Os modelos CLIP são de terceiros e mantêm suas próprias licenças.

