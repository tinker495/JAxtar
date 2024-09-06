<div align="center">
<img src="images/JAxtar.png" alt="logo" width="200"></img>
</div>

# JA<sup>xtar</sup>: Batched parallel A* solver in pure Jax!

JA<sup>xtar</sup> is a project with a JAX-native implementation of parallelizeable A* solver for neural heuristic search research.
This project is inspired by [mctx](https://github.com/google-deepmind/mctx) from google-deepmind. mcts can be written in pure jax, so why not A*?
**The project is not yet fully complete, with many types of puzzles left to be solved, more efficient basic heuristics, learning neural network heuristics, and more.**

mcts, or tree search, is used in many RL algorithmic techniques, starting with AlphaGo, but graph search (not tree search) doesn't seem to have received much attention. Nevertheless, there are puzzle solving algorithms that use neural heuristics like [DeepcubeA](https://github.com/forestagostinelli/DeepCubeA) with A* (graph search).

However, the most frustrating aspect of my brief research(MSc) in this area is the time it takes to pass information back and forth between the GPU and CPU. 
When using neural heuristic as a heuristic to eval a single node, it uses almost 50-80% of the time. Because of this, DeepcubeA batches multiple nodes at the same time, which seems to work quite well.

However, this is not a fundamental solution, and I needed to find a way to remove this bottleneck altogether. This led me to look for ways to perform A* on the GPU, and I found quite a few implementations, but most of them suffer from the following problems.

* They are written in pure c and cuda, which is not compatible with ML research
* They are written in jax or torch, but which are 2d grid environments or connectivity matrices, which cannot scale to an infinite number of different states that cannot all be held in memory
* The implementation itself is dependent on the definition of the state or problem.

To solve this problem, I decided to write code that adheres to the following principles and works.

* Only write in pure Jax
  * For ML research.
* Pure jax priority queue
  * A* needs to have a priority queue because it expends nodes in the order of the node with the smallest sum of heuristic and cost
  * However, the heap used inside python uses list variables, which cannot be jitted, so we need to use a heap that can be iterated over in jax
* A hashable with a state and a hashtable that operates on it.
  * We need this to be able to know if a node in the A* algorithm is closed, open, and what state its parent is in
  * If the state is a simple matter of parsing and indexing, we don't need to hash it, but if it's not and there are nearly infinite states, we need to hash each state to index it and access its unique value
* Everything is batched and parallelised
  * GPUs have a lot of cores, but they are very slow compared to CPUs. To overcome this, algorithms running on GPUs should be written as parallel as possible.

Specially written components in this project include:
* a hash_func_builder for convert defined states to hash keys 
* a hashtable to lookup and insert in a parallel way
* a priority queue that can be batched, pushed and popped
* a fully jitted A* algorithm for puzzles.

This project was a real pain in the arse to write, and I almost felt like I was doing acrobatics with Jax, but I managed to create a fully functional version, and hopefully it will inspire you to stumble upon something amazing when you travel to Jax.

## Result
We can find the optimal path using a jittable, batched A* search as shown below. This is not a blazingly fast result, but it can be used for heuristics using neural networks.
```
Start state
┏━━━┳━━━┳━━━┳━━━┓
┃ C ┃ 1 ┃ B ┃ E ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ A ┃ 5 ┃ 6 ┃   ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ 8 ┃ F ┃ 9 ┃ D ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ 3 ┃ 4 ┃ 2 ┃ 7 ┃
┗━━━┻━━━┻━━━┻━━━┛
Target state
┏━━━┳━━━┳━━━┳━━━┓
┃ 1 ┃ 2 ┃ 3 ┃ 4 ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ 5 ┃ 6 ┃ 7 ┃ 8 ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ 9 ┃ A ┃ B ┃ C ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ D ┃ E ┃ F ┃   ┃
┗━━━┻━━━┻━━━┻━━━┛

JIT compiled
Time:  17.50 seconds
Solution found - 66.0 step
Search states: 8431634 # 481807.65 state per sec
```

```
Vmapped A* search, multiple initial state solution 
Start state
┏━━━┳━━━┳━━━┳━━━┓
┃ 7 ┃ 9 ┃ 3 ┃ C ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ F ┃ 2 ┃ A ┃ D ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ 4 ┃ 8 ┃ 5 ┃   ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ 6 ┃ E ┃ 1 ┃ B ┃
┗━━━┻━━━┻━━━┻━━━┛
.
.
. x 10
Target state
┏━━━┳━━━┳━━━┳━━━┓
┃ 1 ┃ 2 ┃ 3 ┃ 4 ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ 5 ┃ 6 ┃ 7 ┃ 8 ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ 9 ┃ A ┃ B ┃ C ┃
┣━━━╋━━━╋━━━╋━━━┫
┃ D ┃ E ┃ F ┃   ┃
┗━━━┻━━━┻━━━┻━━━┛
vmap astar
# astar_result, solved, solved_idx = jax.vmap(astar_fn, in_axes=(0, 0, None))(states, filled, target)
Time: 104.48 seconds
Solution found [False  True  True  True False  True False  True  True False] # 6/10 in 2e7/10 node at each search
# this means astart_fn is completely vmapable and jitable
```

## Citation
Please use this citation to reference this project.

```bibtex
@software{kyuseokjung2020jaxtar,
  title = {JAxtar: A* solver in pure Jax!},
  author = {Kyuseok Jung},
  url = {https://github.com/tinker495/JAxtar},
  year = {2024},
}
```
