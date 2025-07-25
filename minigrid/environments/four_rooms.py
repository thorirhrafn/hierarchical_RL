
import random

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class FourRoomMazeEnv(MiniGridEnv):
    def __init__(
        self,
        size=13,
        agent_start_pos=(2, 2),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Escape the maze"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        room_x, room_y = 6, 6
        # Generate vertical walls
        self.grid.set(room_x, 1, Wall())
        self.grid.set(room_x, 2, Wall())

        for i in range (4, 10):
           self.grid.set(room_x, i, Wall())
        
        self.grid.set(room_x, 11, Wall())

        #Generate horizontal walls
        self.grid.set(1, room_y, Wall())
        
        for i in range (3, 6):
           self.grid.set(i, room_y, Wall())
        for i in range (7, 9):
           self.grid.set(i, room_y+1, Wall())
        for i in range (10, 12):
           self.grid.set(i, room_y+1, Wall())
        

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 4, height - 4)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Escape the maze"

