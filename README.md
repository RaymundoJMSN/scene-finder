# Scene Finder

App de desktop para Windows que acha mapas de RPG — nos seus arquivos e na internet — a partir de
uma busca só, em português ou inglês.

Ele foi feito porque procurar cena para a sessão significava abrir cinco abas e vasculhar pastas, e
porque as ferramentas prontas que fazem isso demoravam minutos por busca. Aqui a busca local
responde em menos de um décimo de segundo.

![versão](https://img.shields.io/badge/vers%C3%A3o-1.2.0-4ade80) ![licença](https://img.shields.io/badge/licen%C3%A7a-MIT-blue)

## O que ele faz

**Acervo local** — indexa suas pastas de mapas (Foundry VTT ou qualquer outra) e busca por
significado, não por nome de arquivo. Procurar `taverna a noite` acha `Fey Tavern Night` mesmo com a
consulta em português e o arquivo em inglês.

Nos resultados locais: **clique** copia o caminho no formato do Foundry, **shift+clique** abre a
pasta no Explorer e **botão direito** abre a imagem no seu visualizador, em tamanho real. Mapas que
vêm em várias versões (Day/Night/Gridless…) ocupam um card só, com um selo que expande todas — sem
isso, um único mapa tomaria a tela inteira. O botão **≈** procura mapas visualmente parecidos com
aquele.

**Fontes online** — a mesma busca consulta em paralelo o Reddit (r/battlemaps, r/dndmaps,
r/FantasyMaps), o Czepeku (separado em fantasy/scifi × scenes/maps) e criadores que você
configurar. Como esses sites só casam texto em inglês, a consulta é traduzida antes — por um
dicionário embutido de termos de cenário e, quando ele não dá conta da frase, por um tradutor
online com cache. A interface mostra qual termo foi usado de fato. Ainda há botões para abrir Lost
Atlas, JamesRPGArt e Google Imagens.

**Filtros** — em **⚙ Pastas** cada pasta indexada e cada fonte online tem um interruptor. Desligar
uma pasta a esconde da busca **sem apagá-la do índice**: dá para manter uma pasta de mapas
indexada e ligá-la só quando precisar, sem reindexar de novo a cada troca.

## Instalação

Baixe o `SceneFinder-Setup-x.y.z.exe` em [Releases](../../releases) e execute. Instala por usuário,
sem pedir administrador, e cria atalhos no Menu Iniciar e na Área de Trabalho.

Na primeira abertura ele procura sua pasta do Foundry sozinho. Se não achar, ou se seus mapas
estiverem em outro lugar, use **⚙ Pastas** para apontar os diretórios e clicar em salvar — ele
indexa em seguida. Um acervo de ~6.000 imagens leva cerca de 25 minutos de CPU, uma vez só; depois a
indexação é incremental e leva segundos.

O app avisa quando há versão nova, baixa e abre o instalador — suas configurações são preservadas.
Quando uma atualização troca o modelo de busca, o índice antigo deixa de servir e ele se reconstrói
sozinho na primeira abertura.

## Como a busca local funciona

Um modelo de imagem sozinho não resolve battle maps: vistos de cima, quase todos parecem iguais, os
scores empatam e o primeiro resultado vira sorte. O ranking aqui soma três sinais:

1. **Semântica da imagem** — SigLIP2 sobre o conteúdo visual.
2. **Semântica do nome do arquivo** — embedado no mesmo espaço vetorial pelo encoder de texto, que é
   multilíngue. É isso que faz `taverna` alcançar `tavern`, sem tradutor no meio.
3. **Coincidência literal** de palavras do caminho, que vale mais que "parecido".

A escolha do modelo foi medida, não adotada por reputação: `tools/bench_modelos.py` monta consultas
a partir dos nomes dos seus próprios arquivos e mede acerto usando só o sinal visual. Trocar CLIP
ViT-B/32 por SigLIP2 levou o acerto no primeiro resultado de 15% para 25%.

Tudo roda em ONNX no processador. O encoder de texto é quantizado em int8; **o de imagem não** —
quantizá-lo derruba a fidelidade para 0,72 e deixa a busca pior que o modelo antigo. Sem GPU e sem
nuvem: nenhuma imagem sua sai do computador, só o texto da busca vai para as fontes online.

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
venv\Scripts\python indexer.py --check       # pipeline de indexação
venv\Scripts\python ptbr.py                  # tradução das consultas
venv\Scripts\python tools\smoke_encoder.py   # encoder carrega e é multilíngue
venv\Scripts\python tools\verify_onnx.py     # ONNX bate com o modelo original
venv\Scripts\python tools\bench_modelos.py   # compara modelos no seu acervo
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

## Configurando fontes online

As fontes ficam em `config.json` (criado no primeiro uso). Para acompanhar um criador, adicione o
serviço e o ID que aparecem na URL do perfil:

```json
"kemono": [
  { "service": "patreon", "id": "12345678", "name": "Nome do criador" }
]
```

O app não baixa nem redistribui conteúdo de lugar nenhum: ele lista títulos e miniaturas e abre o
post original no navegador. Compre nos criadores que você usa.

## Licença

MIT — veja [LICENSE](LICENSE). Os modelos CLIP são de terceiros e mantêm suas próprias licenças.
