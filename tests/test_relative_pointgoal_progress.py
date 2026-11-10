"""Relative endpoints must not disappear inside a broad arrival disk."""
from types import SimpleNamespace
import numpy as np
import pytest
from tarsphere.harness.core.types import ToolCall
from tarsphere.harness.tools.handlers import NavigateToPointGoalHandler
from tarsphere.infra.perception.online_mapping import OnlineOccupancyGrid

@pytest.mark.parametrize('distance', [0.3, 0.5])
def test_open_space_relative_goal_advances(distance):
    grid = OnlineOccupancyGrid(size_m=6, origin_xy=(-3., -3.))
    grid._observed[:] = True
    grid._ground_free[:] = True
    grid._grid[:] = grid.FREE
    result = NavigateToPointGoalHandler().execute(
        ToolCall(tool='navigate_to_pointgoal', args={
            'forward_m': distance, 'left_m': 0., 'target_description': 'visible floor'}),
        {'observation': SimpleNamespace(pose={'position': [0., 0., 0.], 'yaw_rad': 0.}),
         'online_occupancy_grid': grid})
    assert not result.is_error
    command = result.action['navigation_command']
    assert command['phase'] != 'already_reached'
    assert command['command_body_delta'][0] > 0.1
    assert 'requested distance=' in result.content
    assert 'resolved distance=' in result.content

def test_blocked_region_with_no_progress_reports_no_route(monkeypatch):
    grid = OnlineOccupancyGrid(size_m=6, origin_xy=(-3., -3.))
    grid._grid[:] = grid.FREE
    monkeypatch.setattr(grid, 'astar_with_reason', lambda *a, **kw: (None, 'blocked'))
    monkeypatch.setattr(grid, 'astar_to_goal_region', lambda *a, **kw:
        ([np.zeros(2), np.array([.02, 0.])], np.array([.02, 0.]), 'ok'))
    result = NavigateToPointGoalHandler().execute(
        ToolCall(tool='navigate_to_pointgoal', args={
            'forward_m': .5, 'left_m': 0., 'target_description': 'blocked floor'}),
        {'observation': SimpleNamespace(pose={'position': [0., 0., 0.], 'yaw_rad': 0.}),
         'online_occupancy_grid': grid})
    assert result.metadata['planning_status'] == 'no_route'
    assert result.metadata['planning_reason'] == 'goal_region_no_progress'
    assert 'already_reached' not in result.content
