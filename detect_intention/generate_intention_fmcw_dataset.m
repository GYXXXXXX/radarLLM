function generate_intention_fmcw_dataset(numScenes, outputDir, profile)
% GENERATE_INTENTION_FMCW_DATASET
%
% Generate clean FMCW-MIMO radar scenes with target-intention labels.
%
% Output folder:
%   detect_intention/intention_dataset/
%
% Each scene saves:
%   iq    : [Nfast, Nchirp, Nrx, Nframes] complex single
%   rdMap : [Nrange, Ndoppler, Nframes] single
%   gt    : clean target truth, trajectory, intention, and response labels
%   meta  : scene metadata
%   p     : radar and dataset parameters
%
% This generator intentionally does not simulate interferers or jammers.
%
% Optional usage:
%   generate_intention_fmcw_dataset()
%   generate_intention_fmcw_dataset(10)
%   generate_intention_fmcw_dataset(10, 'E:\tmp\intention_dataset')
%   generate_intention_fmcw_dataset(5000, [], 'compact')

clc;

if nargin < 3 || isempty(profile)
    profile = 'full';
end

p = defaultParams(profile);

if nargin >= 1 && ~isempty(numScenes)
    p.numScenes = numScenes;
end

if nargin >= 2 && ~isempty(outputDir)
    p.outputDir = outputDir;
end

if ~exist(p.outputDir, 'dir')
    mkdir(p.outputDir);
end

if p.cleanOutputDir
    deleteIfExist(fullfile(p.outputDir, 'scene_*.mat'));
    deleteIfExist(fullfile(p.outputDir, 'index.csv'));
end

index_scene_id = zeros(p.numScenes, 1);
index_file = cell(p.numScenes, 1);
index_n_targets = zeros(p.numScenes, 1);
index_intents = cell(p.numScenes, 1);
index_threat_max = zeros(p.numScenes, 1);
index_actions = cell(p.numScenes, 1);
index_all_targets_visible = false(p.numScenes, 1);

fprintf('Generating intention dataset into folder: %s\n', p.outputDir);
fprintf('Profile: %s\n', p.profile);
fprintf('Scenes: %d | frames per scene: %d | Nfast=%d | Nchirp=%d | Nrx=%d | interferers: 0\n\n', ...
    p.numScenes, p.Nframes, p.Nfast, p.Nchirp, p.Nrx);

for sid = 1:p.numScenes
    rng(p.baseSeed + sid);

    [iq, rdMap, gt, meta] = simulateOneScene(p, sid);

    fileName = sprintf('scene_%06d.mat', sid);
    savePath = fullfile(p.outputDir, fileName);
    save(savePath, 'iq', 'rdMap', 'gt', 'meta', 'p', '-v7.3');

    index_scene_id(sid) = sid;
    index_file{sid} = fileName;
    index_n_targets(sid) = meta.nTargets;
    index_intents{sid} = strjoin(meta.intentNames, '|');
    index_threat_max(sid) = meta.maxThreatLevel;
    index_actions{sid} = strjoin(meta.recommendedActions, '|');
    index_all_targets_visible(sid) = meta.allTargetsVisibleAllFrames;

    fprintf('Saved %s | targets = %d | intents = %s | max threat = %d\n', ...
        fileName, meta.nTargets, index_intents{sid}, meta.maxThreatLevel);
end

indexTable = table( ...
    index_scene_id, ...
    index_file, ...
    index_n_targets, ...
    index_intents, ...
    index_threat_max, ...
    index_actions, ...
    index_all_targets_visible, ...
    'VariableNames', { ...
        'scene_id', ...
        'file', ...
        'n_targets', ...
        'intent_names', ...
        'max_threat_level', ...
        'recommended_actions', ...
        'all_targets_visible' ...
    } ...
);

writetable(indexTable, fullfile(p.outputDir, 'index.csv'));

firstScene = fullfile(p.outputDir, 'scene_000001.mat');
visualizeOneScene(firstScene, fullfile(p.outputDir, 'scene_000001_preview.png'));

fprintf('\nDone.\n');
fprintf('Dataset folder: %s\n', p.outputDir);
fprintf('Use visualize_intention_dataset() to create overview figures.\n');

end


function p = defaultParams(profile)

scriptDir = fileparts(mfilename('fullpath'));

p.profile = char(profile);
p.outputDir = fullfile(scriptDir, 'intention_dataset');
p.numScenes = 500;
p.baseSeed = 20260603;
p.cleanOutputDir = true;

% FMCW radar parameters
p.c = 3e8;
p.fc = 77e9;
p.lambda = p.c / p.fc;
p.B = 150e6;
p.Tc = 25e-6;
p.S = p.B / p.Tc;
p.Fs = 10e6;

p.Nfast = 128;
p.Nchirp = 32;
p.Nrx = 8;
p.d = p.lambda / 2;

% Scene parameters
p.frameRate = 10;
p.dt = 1 / p.frameRate;
p.Nframes = 48;

% Radar valid region
p.rMin = 5;
p.rMax = 100;
p.fovDeg = 120;
p.fovRad = deg2rad(p.fovDeg);
p.maxAbsAz = p.fovRad / 2;

% Conservative generation area to keep all targets visible.
p.safeRMin = 18;
p.safeRMax = 88;
p.safeAbsAz = deg2rad(42);
p.requireTargetVisibleAllFrames = true;
p.maxSceneGenerateTry = 1000;
p.maxTargetGenerateTry = 500;

% FFT parameters
p.NrangeFFT = p.Nfast;
p.NdopplerFFT = p.Nchirp;
p.rangeAxis = ((0:p.NrangeFFT-1) / p.NrangeFFT) * p.Fs * p.c / (2 * p.S);
dopplerFreqAxis = ((-p.NdopplerFFT/2):(p.NdopplerFFT/2-1)) / (p.NdopplerFFT * p.Tc);
p.velocityAxis = dopplerFreqAxis * p.lambda / 2;

% Clean target-only signal. Set addThermalNoise=true later if robustness data
% are needed.
p.signalScale = 2.0e4;
p.addThermalNoise = false;
p.snrDb = 35;
p.minNoisePower = 1e-8;

% Intent response reference point. You can regard it as a protected asset,
% route gate, or high-value area in front of the radar.
p.protectedPoint = [72, 0];
p.protectedRadius = 10;

switch lower(p.profile)
    case 'full'
        % Keep the original high-resolution raw-IQ setting.

    case 'compact'
        % About 6x smaller than full:
        % 64 * 16 * 4 * 32 complex single samples per scene.
        p.outputDir = fullfile(scriptDir, 'intention_dataset_compact');
        p.Nfast = 64;
        p.Nchirp = 16;
        p.Nrx = 4;
        p.Nframes = 32;
        p.signalScale = 1.2e4;

    case 'tiny'
        % Very small debug/profile dataset. Use only for fast experiments.
        p.outputDir = fullfile(scriptDir, 'intention_dataset_tiny');
        p.Nfast = 64;
        p.Nchirp = 16;
        p.Nrx = 2;
        p.Nframes = 24;
        p.signalScale = 1.2e4;

    otherwise
        error('Unknown profile: %s. Use full, compact, or tiny.', p.profile);
end

% Recompute axes after a profile changes tensor sizes.
p.NrangeFFT = p.Nfast;
p.NdopplerFFT = p.Nchirp;
p.rangeAxis = ((0:p.NrangeFFT-1) / p.NrangeFFT) * p.Fs * p.c / (2 * p.S);
dopplerFreqAxis = ((-p.NdopplerFFT/2):(p.NdopplerFFT/2-1)) / (p.NdopplerFFT * p.Tc);
p.velocityAxis = dopplerFreqAxis * p.lambda / 2;

end


function [iq, rdMap, gt, meta] = simulateOneScene(p, sid)

nTargets = randi([1, 4]);
targetClassIds = randi([1, 4], 1, nTargets);
intentIds = randi([1, 5], 1, nTargets);

success = false;
targetObjs = emptyObjectArray();

for attempt = 1:p.maxSceneGenerateTry
    targetObjs = generateIntentTargets(p, targetClassIds, intentIds);

    if p.requireTargetVisibleAllFrames
        success = checkTargetsVisibleAllFrames(p, targetObjs);
    else
        success = true;
    end

    if success
        break;
    end
end

if ~success
    error('Scene %d failed to generate visible clean targets after %d tries.', ...
        sid, p.maxSceneGenerateTry);
end

iq = synthesizeIQ(p, targetObjs);
rdMap = computeRDMap(p, iq);
gt = buildGroundTruth(p, targetObjs);

meta.sceneId = sid;
meta.nTargets = nTargets;
meta.nInterferers = 0;
meta.targetClassIds = targetClassIds;
meta.intentIds = intentIds;
meta.intentNames = gt.intentName(:).';
meta.threatLevels = gt.threatLevel(:).';
meta.maxThreatLevel = max(gt.threatLevel);
meta.recommendedActions = unique(gt.recommendedActionName(:).', 'stable');
meta.requireTargetVisibleAllFrames = p.requireTargetVisibleAllFrames;
meta.allTargetsVisibleAllFrames = checkTargetsVisibleAllFrames(p, targetObjs);
meta.llmTaskText = buildScenePrompt(gt);

end


function objs = generateIntentTargets(p, classIds, intentIds)

n = numel(classIds);
objs = emptyObjectArray();

for i = 1:n
    classId = classIds(i);
    intentId = intentIds(i);

    success = false;

    for attempt = 1:p.maxTargetGenerateTry
        obj = makeOneIntentTarget(p, i, classId, intentId);

        if checkTargetsVisibleAllFrames(p, obj)
            success = true;
            break;
        end
    end

    if ~success
        error('Failed to generate target %d with intent %d.', i, intentId);
    end

    objs(end + 1) = obj; %#ok<AGROW>
end

end


function obj = makeOneIntentTarget(p, objId, classId, intentId)

cls = targetClassParam(classId);
intent = intentParam(intentId);

obj.id = objId;
obj.isTarget = true;
obj.targetClassId = classId;
obj.name = cls.name;
obj.rcsDb = cls.rcsDb;
obj.phase0 = 2 * pi * rand();

obj.intentId = intentId;
obj.intentName = intent.name;
obj.intentDescription = intent.description;
obj.threatLevel = intent.threatLevel;
obj.responseActionId = intent.responseActionId;
obj.responseActionName = intent.responseActionName;
obj.intentCue = intent.cue;

[pos, cue] = sampleIntentTrajectory(p, classId, intentId, obj.phase0);
vel = estimateVelocity(pos, p.dt);

obj.pos = pos;
obj.vel = vel;
obj.intentCue = cue;

end


function [pos, cue] = sampleIntentTrajectory(p, classId, intentId, phase0)

T = p.Nframes;
tau = linspace(0, 1, T)';
smoothTau = 3 * tau.^2 - 2 * tau.^3;

switch intentId
    case 1
        % Benign transit: steady motion across the observed sector.
        p0 = samplePolar(randUniform(28, 42), deg2rad(randUniform(-34, -10)));
        p1 = samplePolar(randUniform(62, 82), deg2rad(randUniform(8, 34)));
        cue = 'steady course, no closure toward protected area';

    case 2
        % Approach: range decreases toward radar/guarded water.
        az = deg2rad(randUniform(-28, 28));
        p0 = samplePolar(randUniform(72, 88), az + deg2rad(randUniform(-5, 5)));
        p1 = samplePolar(randUniform(26, 40), az + deg2rad(randUniform(-4, 4)));
        cue = 'closing range toward radar sector';

    case 3
        % Retreat: target moves away after entering the sector.
        az = deg2rad(randUniform(-30, 30));
        p0 = samplePolar(randUniform(24, 38), az + deg2rad(randUniform(-4, 4)));
        p1 = samplePolar(randUniform(66, 86), az + deg2rad(randUniform(-5, 5)));
        cue = 'increasing range and leaving guarded water';

    case 4
        % Loiter/patrol: slow oval or side-to-side motion near one area.
        center = samplePolar(randUniform(42, 62), deg2rad(randUniform(-22, 22)));
        radial = normalizeVec(center);
        lateral = [-radial(2), radial(1)];
        ampA = randUniform(4.0, 8.0);
        ampB = randUniform(1.5, 3.5);
        cycle = randUniform(0.45, 0.80);
        pos = center ...
            + ampA * sin(2*pi*cycle*tau + phase0) .* lateral ...
            + ampB * cos(2*pi*cycle*tau + phase0) .* radial;
        cue = 'repeated local motion near the same area';
        pos = addClassManeuver(pos, classId, tau, phase0);
        return;

    case 5
        % Intercept: high-priority motion toward the protected point.
        startAz = deg2rad(randUniform(-36, 36));
        p0 = samplePolar(randUniform(28, 52), startAz);
        aim = p.protectedPoint + randUniform(-4.0, 4.0) * [0, 1] ...
            + randUniform(-2.0, 2.0) * [1, 0];
        p1 = aim;
        cue = 'fast course toward protected point';

    otherwise
        p0 = samplePolar(30, 0);
        p1 = samplePolar(70, 0);
        cue = 'unknown';
end

baseDir = normalizeVec(p1 - p0);
lateral = [-baseDir(2), baseDir(1)];
curveAmp = targetClassParam(classId).curveAmp;
curve = curveAmp * sin(pi * tau + phase0) .* lateral;
pos = (1 - smoothTau) .* p0 + smoothTau .* p1 + curve;
pos = addClassManeuver(pos, classId, tau, phase0);

end


function pos = addClassManeuver(pos, classId, tau, phase0)

direction = normalizeVec(pos(end, :) - pos(1, :));
if norm(direction) < 1e-6
    radial = normalizeVec(pos(1, :));
    lateral = [-radial(2), radial(1)];
else
    lateral = [-direction(2), direction(1)];
end

switch classId
    case 2
        pos = pos + 0.35 * sin(2*pi*2.0*tau + phase0) .* lateral;
    case 3
        pos = pos + 0.12 * sin(2*pi*5.0*tau + phase0) .* lateral;
    case 4
        pos = pos + 0.65 * sin(2*pi*1.2*tau + phase0) .* lateral;
end

end


function iq = synthesizeIQ(p, objects)

iq = zeros(p.Nfast, p.Nchirp, p.Nrx, p.Nframes, 'single') + ...
     1i * zeros(p.Nfast, p.Nchirp, p.Nrx, p.Nframes, 'single');

tFast = (0:p.Nfast-1)' / p.Fs;
tSlow = (0:p.Nchirp-1) * p.Tc;
rxIdx = 0:p.Nrx-1;

for f = 1:p.Nframes
    frame = zeros(p.Nfast, p.Nchirp, p.Nrx) + ...
            1i * zeros(p.Nfast, p.Nchirp, p.Nrx);

    for k = 1:numel(objects)
        obj = objects(k);
        pos = obj.pos(f, :);
        vel = obj.vel(f, :);

        if ~isVisible(p, pos)
            continue;
        end

        R = norm(pos);
        theta = atan2(pos(2), pos(1));
        radialDir = pos / max(R, eps);
        radialVel = dot(vel, radialDir);

        scats = targetScatterers(obj.targetClassId, f, p, obj.phase0);
        frame = addPointScatterers(frame, R, theta, radialVel, obj.rcsDb, ...
            scats, p, tFast, tSlow, rxIdx);
    end

    if p.addThermalNoise
        frame = addThermalNoise(frame, p);
    end

    iq(:, :, :, f) = single(frame);
end

end


function frame = addPointScatterers(frame, R, theta, radialVel, rcsDb, scats, p, tFast, tSlow, rxIdx)

angleGain = max(cos(theta), 0)^2;
steer = exp(1i * 2*pi * (p.d / p.lambda) * rxIdx * sin(theta));

rcsLin = 10^(rcsDb / 10);
baseAmp = p.signalScale * sqrt(rcsLin) * angleGain / (R^2 + 1);

for s = 1:numel(scats.weight)
    Rs = max(p.rMin, R + scats.dr(s));
    fb = 2 * p.S * Rs / p.c;
    fd = 2 * radialVel / p.lambda + scats.fdOffsetHz(s);

    fastPhase = exp(1i * (2*pi*fb*tFast + scats.phase(s)));
    slowPhase = exp(1i * (2*pi*fd*tSlow));
    sig2d = fastPhase * slowPhase;
    amp = baseAmp * scats.weight(s);

    for rx = 1:p.Nrx
        frame(:, :, rx) = frame(:, :, rx) + amp * sig2d * steer(rx);
    end
end

end


function frame = addThermalNoise(frame, p)

sigPow = mean(abs(frame(:)).^2);
noisePow = max(sigPow / (10^(p.snrDb/10)), p.minNoisePower);
noise = sqrt(noisePow/2) * (randn(size(frame)) + 1i * randn(size(frame)));
frame = frame + noise;

end


function rdMap = computeRDMap(p, iq)

rdMap = zeros(p.NrangeFFT, p.NdopplerFFT, p.Nframes, 'single');
wr = localHann(p.Nfast);
wd = localHann(p.Nchirp).';
win2d = wr * wd;

for f = 1:p.Nframes
    powerRD = zeros(p.NrangeFFT, p.NdopplerFFT);

    for rx = 1:p.Nrx
        x = double(iq(:, :, rx, f));
        x = x .* win2d;
        X = fft(x, p.NrangeFFT, 1);
        X = fftshift(fft(X, p.NdopplerFFT, 2), 2);
        powerRD = powerRD + abs(X).^2;
    end

    powerRD = powerRD / p.Nrx;
    rdMap(:, :, f) = single(10 * log10(powerRD + eps));
end

end


function gt = buildGroundTruth(p, objects)

nObj = numel(objects);

gt.numObjects = nObj;
gt.objId = zeros(nObj, 1);
gt.isTarget = true(nObj, 1);
gt.targetClassId = zeros(nObj, 1);
gt.name = cell(nObj, 1);

gt.intentId = zeros(nObj, 1);
gt.intentName = cell(nObj, 1);
gt.intentDescription = cell(nObj, 1);
gt.intentCue = cell(nObj, 1);
gt.threatLevel = zeros(nObj, 1);
gt.responseActionId = zeros(nObj, 1);
gt.recommendedActionName = cell(nObj, 1);

gt.pos = zeros(nObj, p.Nframes, 2, 'single');
gt.vel = zeros(nObj, p.Nframes, 2, 'single');
gt.range = zeros(nObj, p.Nframes, 'single');
gt.azimuth = zeros(nObj, p.Nframes, 'single');
gt.radialVel = zeros(nObj, p.Nframes, 'single');
gt.distanceToProtectedPoint = zeros(nObj, p.Nframes, 'single');
gt.visible = false(nObj, p.Nframes);

for k = 1:nObj
    obj = objects(k);

    gt.objId(k) = obj.id;
    gt.targetClassId(k) = obj.targetClassId;
    gt.name{k} = obj.name;
    gt.intentId(k) = obj.intentId;
    gt.intentName{k} = obj.intentName;
    gt.intentDescription{k} = obj.intentDescription;
    gt.intentCue{k} = obj.intentCue;
    gt.threatLevel(k) = obj.threatLevel;
    gt.responseActionId(k) = obj.responseActionId;
    gt.recommendedActionName{k} = obj.responseActionName;

    for f = 1:p.Nframes
        pos = obj.pos(f, :);
        vel = obj.vel(f, :);

        R = norm(pos);
        az = atan2(pos(2), pos(1));
        radialDir = pos / max(R, eps);
        vr = dot(vel, radialDir);

        gt.pos(k, f, :) = single(pos);
        gt.vel(k, f, :) = single(vel);
        gt.range(k, f) = single(R);
        gt.azimuth(k, f) = single(az);
        gt.radialVel(k, f) = single(vr);
        gt.distanceToProtectedPoint(k, f) = single(norm(pos - p.protectedPoint));
        gt.visible(k, f) = isVisible(p, pos);
    end
end

gt.targetObjectIndices = find(gt.isTarget);
gt.interferenceObjectIndices = [];

end


function scats = targetScatterers(classId, frameIdx, p, phase0)

t = (frameIdx - 1) * p.dt;

switch classId
    case 1
        scats.weight = [1.00, 0.65, 0.45];
        scats.dr = [0.0, 1.2, -1.1];
        scats.fdOffsetHz = [0, 0, 0];
        scats.phase = phase0 + [0, 1.7, 3.1];

    case 2
        rotor = 140 + 20 * sin(2*pi*0.2*t);
        scats.weight = [1.00, 0.35, 0.35, 0.20, 0.20];
        scats.dr = [0, 0.05, -0.05, 0.1, -0.1];
        scats.fdOffsetHz = [0, rotor, -rotor, 2*rotor, -2*rotor];
        scats.phase = phase0 + [0, 0.8, 1.4, 2.1, 2.8];

    case 3
        gait = 45 + 10 * sin(2*pi*1.3*t);
        wLeg = 0.30 + 0.08 * sin(2*pi*1.3*t);
        scats.weight = [1.00, wLeg, wLeg];
        scats.dr = [0, 0.25, -0.25];
        scats.fdOffsetHz = [0, gait, -gait];
        scats.phase = phase0 + [0, 1.1, 2.2];

    case 4
        scats.weight = [1.00, 0.25, 0.18];
        scats.dr = [0, 0.35, -0.25];
        scats.fdOffsetHz = [0, 30 * sin(2*pi*0.4*t), -25 * cos(2*pi*0.3*t)];
        scats.phase = phase0 + [0, 2.5, 4.1];

    otherwise
        scats.weight = 1;
        scats.dr = 0;
        scats.fdOffsetHz = 0;
        scats.phase = phase0;
end

end


function cls = targetClassParam(classId)

switch classId
    case 1
        cls.name = 'T1_slow_smooth';
        cls.rcsDb = 12;
        cls.curveAmp = 0.25;

    case 2
        cls.name = 'T2_agile';
        cls.rcsDb = -2;
        cls.curveAmp = 0.70;

    case 3
        cls.name = 'T3_small_slow';
        cls.rcsDb = 0;
        cls.curveAmp = 0.18;

    case 4
        cls.name = 'T4_fast_maneuver';
        cls.rcsDb = 6;
        cls.curveAmp = 1.00;

    otherwise
        error('Unknown target class id.');
end

end


function intent = intentParam(intentId)

switch intentId
    case 1
        intent.name = 'benign_transit';
        intent.description = 'Target passes through the sector without closing on the protected point.';
        intent.cue = 'stable track and low-risk geometry';
        intent.threatLevel = 1;
        intent.responseActionId = 1;
        intent.responseActionName = 'monitor';

    case 2
        intent.name = 'approach';
        intent.description = 'Target closes range toward the radar or guarded water.';
        intent.cue = 'range decreases over time';
        intent.threatLevel = 3;
        intent.responseActionId = 2;
        intent.responseActionName = 'increase_tracking_rate';

    case 3
        intent.name = 'retreat';
        intent.description = 'Target moves away from the radar or guarded water.';
        intent.cue = 'range increases over time';
        intent.threatLevel = 1;
        intent.responseActionId = 1;
        intent.responseActionName = 'monitor';

    case 4
        intent.name = 'loiter_patrol';
        intent.description = 'Target remains near one area with repeated local motion.';
        intent.cue = 'low displacement with recurring turns';
        intent.threatLevel = 2;
        intent.responseActionId = 3;
        intent.responseActionName = 'classify_and_shadow';

    case 5
        intent.name = 'intercept';
        intent.description = 'Target heads toward the protected point or route gate.';
        intent.cue = 'distance to protected point decreases quickly';
        intent.threatLevel = 4;
        intent.responseActionId = 4;
        intent.responseActionName = 'alert_and_allocate_tracker';

    otherwise
        error('Unknown intent id.');
end

end


function text = buildScenePrompt(gt)

lines = cell(gt.numObjects + 1, 1);
lines{1} = 'Infer each target intention from clean radar trajectory history.';

for k = 1:gt.numObjects
    lines{k + 1} = sprintf( ...
        'Target %d: class=%s, intent=%s, threat=%d, action=%s.', ...
        gt.objId(k), gt.name{k}, gt.intentName{k}, gt.threatLevel(k), ...
        gt.recommendedActionName{k});
end

text = strjoin(lines, newline);

end


function flag = checkTargetsVisibleAllFrames(p, targetObjs)

flag = true;

for k = 1:numel(targetObjs)
    obj = targetObjs(k);

    for f = 1:p.Nframes
        if ~isVisible(p, obj.pos(f, :))
            flag = false;
            return;
        end
    end
end

end


function flag = isVisible(p, pos)

x = pos(1);
y = pos(2);
R = sqrt(x^2 + y^2);
az = atan2(y, x);

flag = (x > 0) && ...
    (R >= p.rMin) && ...
    (R <= p.rMax) && ...
    (abs(az) <= p.maxAbsAz);

end


function vel = estimateVelocity(pos, dt)

T = size(pos, 1);
vel = zeros(T, 2);

if T == 1
    return;
end

vel(1, :) = (pos(2, :) - pos(1, :)) / dt;
vel(T, :) = (pos(T, :) - pos(T-1, :)) / dt;

for i = 2:T-1
    vel(i, :) = (pos(i+1, :) - pos(i-1, :)) / (2 * dt);
end

end


function point = samplePolar(r, az)

point = [r * cos(az), r * sin(az)];

end


function y = randUniform(a, b)

y = a + (b - a) * rand();

end


function v = normalizeVec(v)

n = norm(v);

if n < 1e-9
    v = [1, 0];
else
    v = v / n;
end

end


function w = localHann(n)

if n <= 1
    w = ones(n, 1);
else
    idx = (0:n-1)';
    w = 0.5 - 0.5 * cos(2*pi*idx/(n-1));
end

end


function objs = emptyObjectArray()

objs = struct( ...
    'id', {}, ...
    'isTarget', {}, ...
    'targetClassId', {}, ...
    'name', {}, ...
    'rcsDb', {}, ...
    'phase0', {}, ...
    'intentId', {}, ...
    'intentName', {}, ...
    'intentDescription', {}, ...
    'threatLevel', {}, ...
    'responseActionId', {}, ...
    'responseActionName', {}, ...
    'intentCue', {}, ...
    'pos', {}, ...
    'vel', {} ...
);

end


function deleteIfExist(pattern)

files = dir(pattern);

for i = 1:numel(files)
    fullPath = fullfile(files(i).folder, files(i).name);

    if exist(fullPath, 'file')
        delete(fullPath);
    end
end

end


function visualizeOneScene(sceneFile, savePath)

load(sceneFile, 'rdMap', 'gt', 'p', 'meta');

fig = figure('Visible', 'off', 'Name', 'Intention Scene Preview', 'Color', 'w');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
hold on;
grid on;
axis equal;
drawFOV(p);
plot(p.protectedPoint(1), p.protectedPoint(2), 'mp', ...
    'MarkerFaceColor', 'm', 'MarkerSize', 12);
theta = linspace(0, 2*pi, 120);
plot(p.protectedPoint(1) + p.protectedRadius * cos(theta), ...
     p.protectedPoint(2) + p.protectedRadius * sin(theta), ...
     'm:', 'LineWidth', 1.0);

for k = 1:gt.numObjects
    xy = squeeze(gt.pos(k, :, :));
    plot(xy(:, 1), xy(:, 2), '-o', 'LineWidth', 1.5, 'MarkerSize', 3);
    plot(xy(1, 1), xy(1, 2), 'go', 'MarkerFaceColor', 'g');
    plot(xy(end, 1), xy(end, 2), 'ks', 'MarkerFaceColor', 'k');
    text(xy(1, 1), xy(1, 2), ...
        sprintf('  T%d %s', k, gt.intentName{k}), ...
        'Interpreter', 'none', 'FontSize', 8);
end

xlabel('x / m');
ylabel('y / m');
title(sprintf('Scene %d Clean Targets with Intent', meta.sceneId), ...
    'Interpreter', 'none');
xlim([0, p.rMax + 5]);
ylim([-p.rMax * sind(p.fovDeg / 2) - 5, ...
       p.rMax * sind(p.fovDeg / 2) + 5]);

nexttile;
frameId = p.Nframes;
imagesc(p.velocityAxis, p.rangeAxis, rdMap(:, :, frameId));
axis xy;
xlabel('Radial velocity / m/s');
ylabel('Range / m');
title(sprintf('Range-Doppler | Frame %d', frameId));
colorbar;

sgtitle(sprintf('Intents: %s', strjoin(gt.intentName(:).', ', ')), ...
    'Interpreter', 'none');

try
    exportgraphics(fig, savePath, 'Resolution', 200);
catch
    saveas(fig, savePath);
end

close(fig);

end


function drawFOV(p)

theta = linspace(-p.maxAbsAz, p.maxAbsAz, 200);
xOuter = p.rMax * cos(theta);
yOuter = p.rMax * sin(theta);
xInner = p.rMin * cos(theta);
yInner = p.rMin * sin(theta);

plot(xOuter, yOuter, 'k--', 'LineWidth', 1.0);
plot(xInner, yInner, 'k--', 'LineWidth', 1.0);
plot([p.rMin*cos(-p.maxAbsAz), p.rMax*cos(-p.maxAbsAz)], ...
     [p.rMin*sin(-p.maxAbsAz), p.rMax*sin(-p.maxAbsAz)], ...
     'k--', 'LineWidth', 1.0);
plot([p.rMin*cos(p.maxAbsAz), p.rMax*cos(p.maxAbsAz)], ...
     [p.rMin*sin(p.maxAbsAz), p.rMax*sin(p.maxAbsAz)], ...
     'k--', 'LineWidth', 1.0);
plot(0, 0, 'rp', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
text(0, 0, ' Radar', 'FontSize', 8);

end
