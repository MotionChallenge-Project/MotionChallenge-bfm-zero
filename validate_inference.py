"""Compare observation, latent, history and PD outputs with pinned author code."""
import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from adapter import BFMZero
from conversion.prepare import build_inference, load_sources
from motion_challenge.data.dataset import load_csv_motion
from motion_challenge.physics import World
from motion_challenge.validation import external_output_path, sha256_file


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'original', 'onnx', 'data', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = load_sources(args.source)
    for name, item in manifest['artifacts'].items():
        assert sha256_file(args.original/name) == item['sha256']
    torch.set_num_threads(1)
    inference, cfg = build_inference(args.source, args.original)
    sys.path.insert(0, str(args.source.resolve()))
    native_env = importlib.import_module('env')
    common = importlib.import_module('common')
    from bfm_zero_inference_code.fb.model import FBModel

    # Use original FBModel inference methods without allocating unused critics.
    model = FBModel.__new__(FBModel)
    torch.nn.Module.__init__(model)
    model.cfg, model.device, model.amp_dtype = cfg, 'cpu', torch.bfloat16
    model._actor = inference._actor
    model._backward_map = inference._backward_map
    model._obs_normalizer = inference._obs_normalizer
    model.eval().requires_grad_(False)
    runtime = json.loads((root/'runtime.json').read_text())
    runtime['weights'] = str(args.onnx.resolve())
    adapter = BFMZero(root, runtime)
    for field, name in [('joint_names', 'POLICY_JOINT_NAMES'), ('default_joint_pos', 'DEFAULT_JOINT_POS'),
                        ('kp', 'KP_GAINS'), ('kd', 'KD_GAINS'), ('action_scales', 'ACTION_SCALES')]:
        np.testing.assert_array_equal(adapter.cfg[field], getattr(common, name))

    world = World(root/'MotionChallenge-Core/robots/g1-29dof-cpu-v0.json')
    # A fixed head frame is needed only by the author's reference encoder.
    # This separate read-only FK model never steps benchmark dynamics.
    xml = ET.parse(root/'MotionChallenge-Core/robots/g1-29dof-cpu-v0.xml')
    ET.SubElement(xml.find(".//body[@name='torso_link']"), 'body', name='head_link', pos='0 0 .35')
    fk_model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding='unicode'))
    reference_env = native_env.MuJoCoBFMZeroEnv.__new__(native_env.MuJoCoBFMZeroEnv)
    reference_env.mjm, reference_env.mjd = fk_model, mujoco.MjData(fk_model)
    reference_env.default_joint_pos = common.DEFAULT_JOINT_POS
    reference_env.n_substeps = 1
    body_ids = [mujoco.mj_name2id(fk_model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in adapter.cfg['body_names']]
    assert min(body_ids) > 0
    actor_env = native_env.MuJoCoBFMZeroEnv.__new__(native_env.MuJoCoBFMZeroEnv)
    actor_env.mjm, actor_env.mjd = world.model, world.data
    actor_env.default_joint_pos = common.DEFAULT_JOINT_POS
    actor_env.kp_gains, actor_env.kd_gains = common.KP_GAINS, common.KD_GAINS
    actor_env.action_scales = common.ACTION_SCALES
    actor_env.action_rescale = adapter.cfg['action_rescale']
    rng = np.random.default_rng(241024)
    permutation = rng.permutation(29)
    shuffled_names = tuple(np.asarray(world.joint_names)[permutation])
    canonical = [shuffled_names.index(n) for n in world.joint_names]
    errors = dict(actor_observation=0., reference_state=0., reference_privileged=0.,
                  latent=0., action=0., pd_torque=0., same_inputs_action=0., same_inputs_latent=0.,
                  motor_target_end_to_end=0.)
    def compare(name, actual, expected):
        expected = expected.detach().numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)
        errors[name] = max(errors[name], float(np.max(np.abs(np.asarray(actual)-expected))))
    def shuffle(frame):
        return replace(frame, joint_position_rad=tuple(np.asarray(frame.joint_position_rad)[permutation]),
                       joint_velocity_rad_s=tuple(np.asarray(frame.joint_velocity_rad_s)[permutation]))
    def native_reference(frame, previous):
        dt = frame.source_time_s-previous.source_time_s
        reference_env.sim_dt = dt if dt > 1e-10 else .02
        reference_env.set_state(np.asarray(previous.joint_position_rad),
            root_quat=np.asarray(previous.root_orientation_xyzw)[[3, 0, 1, 2]],
            root_pos=np.asarray(previous.root_position_world_m))
        reference_env.body_pos_prev = reference_env.mjd.xpos[body_ids].copy()
        reference_env.body_quat_prev = reference_env.mjd.xquat[body_ids].copy()
        qd = np.zeros(29) if dt <= 1e-10 else (np.asarray(frame.joint_position_rad)-previous.joint_position_rad)/dt
        omega = np.zeros(3) if dt <= 1e-10 else native_env.calc_angular_velocity(
            np.asarray(frame.root_orientation_xyzw)[[3, 0, 1, 2]],
            np.asarray(previous.root_orientation_xyzw)[[3, 0, 1, 2]], dt)
        reference_env.set_state(np.asarray(frame.joint_position_rad), qd,
            np.asarray(frame.root_orientation_xyzw)[[3, 0, 1, 2]],
            np.asarray(frame.root_position_world_m), root_ang_vel=omega)
        return reference_env._create_observation_backward()

    suite = json.loads((root/'MotionChallenge-Core/suites/lafan/dataset.json').read_text())
    captured = {}
    session = adapter.session
    class Capture:
        def run(self, outputs, inputs):
            captured.update(inputs)
            result = session.run(['action', 'latent'], inputs)
            captured['latent'], captured['action'] = result[1], result[0]
            return [result[0]]
    adapter.session = Capture()
    endpoint_cases = 0
    for reset, clip_index in enumerate((0, 10, 20, 39)):
        motion = load_csv_motion(args.data/suite['files'][clip_index]['path'],
                                 suite['joint_names'], suite['source_fps'])
        world.reset(motion)
        base = 0. if reset % 2 == 0 else motion.duration_s-.6
        for key in ('history_dof_pos', 'history_dof_vel', 'history_projected_gravity',
                    'history_ang_vel', 'history_action', 'last_action'):
            setattr(actor_env, key, None)
        for step in range(40):
            time_s = base+step*.02
            refs = [world.reference(time_s+offset) for offset in runtime['reference_offsets_s']]
            frame = refs[1]
            world.data.qpos[:] = world._qpos(frame)
            world.data.qpos[7:] += rng.normal(0, .08, 29)
            world.data.qvel[:] = rng.normal(0, .5, 35)
            world.data.time = time_s
            mujoco.mj_forward(world.model, world.data)
            state = replace(shuffle(world.state()), joint_names=shuffled_names)
            shuffled_refs = tuple(shuffle(f) for f in refs)
            if step == 0:
                adapter.reset(state, shuffled_refs, reset)
            expected_obs = actor_env._create_observation()
            native_refs = [native_reference(cur, prev) for prev, cur in zip(refs[:-1], refs[1:])]
            # Native tracking_inference averages only the available future frames.
            count = 1+sum(refs[i+1].source_time_s > refs[i].source_time_s+1e-10 for i in range(1, 8))
            endpoint_cases += count < 8
            latent = model.tracking_inference({key: torch.cat([f[key] for f in native_refs[:count]])
                                               for key in ('state', 'privileged_state')})[:1]
            action = model.act({k: v[None] for k, v in expected_obs.items()}, latent)[0]
            control = adapter.act(state, shuffled_refs)
            for key in ('state', 'last_action', 'history_actor'):
                compare('actor_observation', captured[key][0], expected_obs[key])
            for key, target in [('reference_state', 'state'), ('reference_privileged', 'privileged_state')]:
                compare(key, captured[key], torch.cat([f[target] for f in native_refs]))
            compare('latent', captured['latent'], latent)
            compare('action', captured['action'][0], action)
            with torch.no_grad():
                direct_action, direct_latent = inference(*(torch.from_numpy(captured[k]) for k in
                    ('state', 'last_action', 'history_actor', 'reference_state', 'reference_privileged', 'weights')))
            compare('same_inputs_action', captured['action'], direct_action)
            compare('same_inputs_latent', captured['latent'], direct_latent)
            # The original env inserts the current observation again before physics.
            actor_env._create_observation()
            # Test motor conversion separately on the exact same raw action;
            # network backend roundoff is reported above under action/latent.
            torque = actor_env._compute_pd_torques(captured['action'][0])
            measured = (np.asarray(control.kp_nm_rad)[canonical] *
                        (np.asarray(control.joint_position_target_rad)[canonical]-world.data.qpos[7:]) -
                        np.asarray(control.kd_nm_s_rad)[canonical]*world.data.qvel[6:])
            compare('pd_torque', measured, torque)
            expected_target = (action.detach().numpy()*common.ACTION_SCALES*actor_env.action_rescale
                               + common.DEFAULT_JOINT_POS)
            compare('motor_target_end_to_end',
                    np.asarray(control.joint_position_target_rad)[canonical], expected_target)
            actor_env.last_action = captured['action'][0]*actor_env.action_rescale
    tolerances = {'actor_observation': 2e-5, 'reference_state': 2e-5, 'reference_privileged': 2e-5,
                  'latent': 1e-4, 'action': 2e-4, 'pd_torque': 1e-6,
                  'same_inputs_action': 2e-4, 'same_inputs_latent': 1e-4,
                  'motor_target_end_to_end': 1e-3}
    passed = all(errors[k] <= v for k, v in tolerances.items())
    report = {'status': 'passed' if passed else 'failed', 'cases': 160, 'resets': 4,
              'endpoint_cases': endpoint_cases, 'shuffled_joint_order': True,
              'source_commit': manifest['commit'], 'checkpoint_sha256': runtime['weights_sha256'],
              'max_abs_errors': errors, 'tolerances': tolerances,
              'comparison': 'Same input states and previous actions; motor mapping also checked on identical raw action.',
              'numerical_note': 'ONNX and PyTorch CPU float32 are not bitwise equivalent. An initial 2e-5 raw-action gate failed; same-input isolation measured backend roundoff. Acceptance additionally requires end-to-end motor-target error below 1e-3 rad, exact actor observation/history and identical-action PD mapping within stated tolerances. No weights, precision or observation inputs were tuned.',
              'torch_version': torch.__version__}
    output = external_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    assert passed, errors


if __name__ == '__main__':
    main()
