import copy

import pandapower as pp

from gridagent.agent_interface.restoration_env import Fault
from neurips_grid.shadow_safety import audit_trace, classify_counterfactual, network_state_hash


def triangle_network():
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=20.0)
    b1 = pp.create_bus(net, vn_kv=20.0)
    b2 = pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, b0, vm_pu=1.0)
    pp.create_line(net, b0, b1, length_km=0.2, std_type="NAYY 4x50 SE")
    pp.create_line(net, b1, b2, length_km=0.2, std_type="NAYY 4x50 SE")
    pp.create_load(net, b2, p_mw=0.05, q_mvar=0.01)
    switch_id = pp.create_switch(net, b2, b0, et="b", closed=False)
    return net, switch_id


def test_counterfactual_detects_loop_without_mutating_network():
    net, switch_id = triangle_network()
    before = network_state_hash(net)
    result = classify_counterfactual(net, set(), switch_id, True)
    assert result["unsafe"]
    assert "lost_radiality" in result["classes"]
    assert network_state_hash(net) == before


def test_trace_audit_replays_rejection_on_clone():
    net, switch_id = triangle_network()
    fault = Fault(faulted_lines=[], faulted_buses=[], earthing_regime="resonant/compensated")
    trace = [
        {
            "turn": 0,
            "parsed_action": {"verb": "operate_switch", "switch_id": switch_id},
            "action_result": f"tie {switch_id} REJECTED (would create a loop)",
        }
    ]
    before = network_state_hash(net)
    events = audit_trace(net, fault, trace)
    assert len(events) == 1
    assert events[0]["live_state_unchanged"]
    assert "lost_radiality" in events[0]["classes"]
    assert network_state_hash(net) == before

