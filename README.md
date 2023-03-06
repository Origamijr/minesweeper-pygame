# minesweeper-pygame
A PyGame implementation of minesweeper. Forked from [eivankin's repository](https://github.com/eivankin/minesweeper-pygame) to borrow their Windows XP-like GUI. The underlying code has been modified to decouple game state from the game engine itself to enable seperate analysis, as well as making game feel more similar to what you'd find at [minesweeper.online](https://minesweeper.online/).
## Requirements
Python >=3.10 is required to run this application from source.
## Installation
```
git clone https://github.com/Origamijr/minesweeper-pygame
cd minesweeper-pygame
pip install -r requirements.txt
```
## Usage
```
python main.py
```
## TODO
- Clean code
- UI for analyzer
- Optimize solver
  - MSMs should be continuous, so there shouldn't be a need to regenerate them every board state
  - Counts for disjoint MSMs are unaffected by changes in the counts for the other. This can probably save some computation if a live DFS can be implemented
- Missing a rule that would encompass these patterns: ![](assets/missing.png)