from dissertation.model import World
from dissertation.view import WorldView

if __name__ == "__main__":
    world = World(100, 100, 1)
    view = WorldView(world, tile_size=10)
    view.run()