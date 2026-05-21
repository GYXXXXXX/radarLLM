function generate_fmcw_traj_dataset()
% GENERATE_FMCW_TRAJ_DATASET
%
% 基于 FMCW-MIMO 雷达的二维连续场景仿真数据集。
%
% 当前版本设计：
%   1. 每个场景直接生成连续 48 帧；
%   2. 数据生成阶段不写死前 32 帧 / 后 16 帧；
%   3. 所有真实目标在 48 帧内必须都处于雷达可测范围；
%   4. 干扰物不强制全程可见；
%   5. 保存 IQ 数据、Range-Doppler 图、轨迹真值和可见性 mask。
%
% 每个 scene 保存：
%   iq    : [Nfast, Nchirp, Nrx, Nframes] complex single
%   rdMap : [Nrange, Ndoppler, Nframes] single
%   gt    : 轨迹真值和标签
%   meta  : 场景元信息
%   p     : 雷达参数和数据集参数

clc;
clear;

p = defaultParams();

if ~exist(p.outputDir, 'dir')
    mkdir(p.outputDir);
end

if p.cleanOutputDir
    delete(fullfile(p.outputDir, 'scene_*.mat'));
    deleteIfExist(fullfile(p.outputDir, 'index.csv'));
end

index_scene_id = zeros(p.numScenes, 1);
index_file = cell(p.numScenes, 1);
index_n_targets = zeros(p.numScenes, 1);
index_n_interferers = zeros(p.numScenes, 1);
index_relation = cell(p.numScenes, 1);
index_all_targets_visible = false(p.numScenes, 1);

fprintf('Generating dataset into folder: %s\n', p.outputDir);
fprintf('Each scene has %d continuous frames.\n\n', p.Nframes);

for sid = 1:p.numScenes
    rng(p.baseSeed + sid);

    [iq, rdMap, gt, meta] = simulateOneScene(p, sid);

    fileName = sprintf('scene_%06d.mat', sid);
    savePath = fullfile(p.outputDir, fileName);

    save(savePath, 'iq', 'rdMap', 'gt', 'meta', 'p', '-v7.3');

    index_scene_id(sid) = sid;
    index_file{sid} = fileName;
    index_n_targets(sid) = meta.nTargets;
    index_n_interferers(sid) = meta.nInterferers;
    index_relation{sid} = meta.relationName;
    index_all_targets_visible(sid) = meta.allTargetsVisibleAllFrames;

    fprintf('Saved %s | targets = %d | interferers = %d | relation = %s | all targets visible = %d\n', ...
        fileName, meta.nTargets, meta.nInterferers, meta.relationName, meta.allTargetsVisibleAllFrames);
end

indexTable = table( ...
    index_scene_id, ...
    index_file, ...
    index_n_targets, ...
    index_n_interferers, ...
    index_relation, ...
    index_all_targets_visible, ...
    'VariableNames', { ...
        'scene_id', ...
        'file', ...
        'n_targets', ...
        'n_interferers', ...
        'relation', ...
        'all_targets_visible' ...
    } ...
);

writetable(indexTable, fullfile(p.outputDir, 'index.csv'));

firstScene = fullfile(p.outputDir, 'scene_000001.mat');
visualizeScene(firstScene);

fprintf('\nDone.\n');
fprintf('Generated %d scenes.\n', p.numScenes);
fprintf('Each scene has %d continuous frames.\n', p.Nframes);
fprintf('\nTraining split should be done later, for example:\n');
fprintf('  inputFrames = 1:32;\n');
fprintf('  predFrames  = 33:48;\n');

end


function p = defaultParams()
% 参数设置

p.outputDir = 'fmcw_traj_dataset_1000scenes';
p.numScenes = 2000;
p.baseSeed = 1202;
p.cleanOutputDir = true;

% =========================
% FMCW radar parameters
% =========================
p.c = 3e8;
p.fc = 77e9;
p.lambda = p.c / p.fc;

p.B = 150e6;              % FMCW bandwidth
p.Tc = 25e-6;             % chirp duration
p.S = p.B / p.Tc;         % chirp slope
p.Fs = 10e6;              % ADC sampling frequency

p.Nfast = 128;            % fast-time samples per chirp
p.Nchirp = 32;            % chirps per frame
p.Nrx = 8;                % virtual RX antennas
p.d = p.lambda / 2;       % ULA spacing

% =========================
% Scene parameters
% =========================
p.frameRate = 10;
p.dt = 1 / p.frameRate;
p.Nframes = 48;

% 注意：
% 数据生成阶段只生成连续 48 帧。
% 前 32 帧作为输入、后 16 帧作为预测目标，是训练阶段再处理。

% Radar valid region
p.rMin = 5;
p.rMax = 100;
p.fovDeg = 120;
p.fovRad = deg2rad(p.fovDeg);
p.maxAbsAz = p.fovRad / 2;

% 为了保证目标 48 帧内都可见，生成时使用更保守的安全区域
p.safeRMin = 18;
p.safeRMax = 80;
p.safeAbsAz = deg2rad(42);

% 是否要求所有真实目标 48 帧内都可见
p.requireTargetVisibleAllFrames = true;
p.maxSceneGenerateTry = 1000;

% FFT parameters
p.NrangeFFT = p.Nfast;
p.NdopplerFFT = p.Nchirp;

p.rangeAxis = ((0:p.NrangeFFT-1) / p.NrangeFFT) * p.Fs * p.c / (2 * p.S);
dopplerFreqAxis = ((-p.NdopplerFFT/2):(p.NdopplerFFT/2-1)) / (p.NdopplerFFT * p.Tc);
p.velocityAxis = dopplerFreqAxis * p.lambda / 2;

% Signal strength
p.signalScale = 2.0e4;
p.jammerScale = 0.8;
p.snrDb = 12;
p.minNoisePower = 1e-5;

end


function [iq, rdMap, gt, meta] = simulateOneScene(p, sid)

nTargets = randi([1, 4]);
nInterferers = randi([0, 2]);

targetClassIds = randperm(4, nTargets);
interfClassIds = randperm(2, nInterferers);

relationId = randi([1, 4]);
relationNames = {'formation', 'crossing', 'leader_follower', 'converging'};
relationName = relationNames{relationId};

success = false;
targetObjs = emptyObjectArray();

for attempt = 1:p.maxSceneGenerateTry
    targetObjs = generateLinkedTargets(p, targetClassIds, relationId);

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
    error('Scene %d failed to generate fully visible targets after %d tries.', ...
        sid, p.maxSceneGenerateTry);
end

interfObjs = generateInterferers(p, interfClassIds, numel(targetObjs) + 1);

if isempty(interfObjs)
    objects = targetObjs;
else
    objects = [targetObjs, interfObjs];
end

iq = synthesizeIQ(p, objects);
rdMap = computeRDMap(p, iq);
gt = buildGroundTruth(p, objects);

meta.sceneId = sid;
meta.nTargets = nTargets;
meta.nInterferers = nInterferers;
meta.relationId = relationId;
meta.relationName = relationName;
meta.targetClassIds = targetClassIds;
meta.interferenceClassIds = interfClassIds;
meta.requireTargetVisibleAllFrames = p.requireTargetVisibleAllFrames;
meta.allTargetsVisibleAllFrames = checkTargetsVisibleAllFrames(p, targetObjs);

end


function objs = generateLinkedTargets(p, classIds, relationId)
% 生成有联系的目标轨迹。
%
% 四种关系：
%   1. formation       : 编队运动
%   2. crossing        : 交叉运动
%   3. leader_follower : 跟随运动
%   4. converging      : 汇聚运动

n = numel(classIds);
T = p.Nframes;

tau = linspace(0, 1, T)';
smoothTau = 3 * tau.^2 - 2 * tau.^3;

objs = emptyObjectArray();

[centerStart, centerEnd] = sampleSafeMotionSegment(p);
baseDir = normalizeVec(centerEnd - centerStart);
latDir = [-baseDir(2), baseDir(1)];

for i = 1:n
    classId = classIds(i);
    cls = targetClassParam(classId);

    obj.id = i;
    obj.isTarget = true;
    obj.targetClassId = classId;
    obj.interferenceClassId = 0;
    obj.name = cls.name;
    obj.rcsDb = cls.rcsDb;
    obj.phase0 = 2 * pi * rand();

    offsetScale = 2.5;

    switch relationId
        case 1
            % formation: 保持相对队形
            alongOffset = -(i - 1) * randUniform(2.0, 4.5);
            lateralOffset = (i - (n + 1) / 2) * randUniform(2.0, 4.0);
            offset = alongOffset * baseDir + lateralOffset * latDir;

            p0 = centerStart + offset;
            p1 = centerEnd + offset;

        case 2
            % crossing: 横向位置交换，形成交叉运动
            lateralOffset = (i - (n + 1) / 2) * randUniform(5.0, 8.0);
            alongJitter = randUniform(-2.0, 2.0);

            p0 = centerStart + alongJitter * baseDir + lateralOffset * latDir;
            p1 = centerEnd + alongJitter * baseDir - lateralOffset * latDir;

        case 3
            % leader-follower: 沿着同一方向前后跟随
            alongOffset = -(i - 1) * randUniform(4.0, 7.0);
            lateralOffset = (i - (n + 1) / 2) * randUniform(0.5, 1.5);
            offset = alongOffset * baseDir + lateralOffset * latDir;

            p0 = centerStart + offset;
            p1 = centerEnd + offset;

        case 4
            % converging: 从不同初始位置向共同区域汇聚
            lateralOffset = (i - (n + 1) / 2) * randUniform(5.0, 9.0);
            alongOffset = randUniform(-4.0, 4.0);

            p0 = centerStart + alongOffset * baseDir + lateralOffset * latDir;
            p1 = centerEnd + randUniform(-offsetScale, offsetScale) * latDir ...
                          + randUniform(-offsetScale, offsetScale) * baseDir;

        otherwise
            p0 = centerStart;
            p1 = centerEnd;
    end

    curveAmp = cls.curveAmp;
    curvePhase = obj.phase0;

    curve = curveAmp * sin(pi * tau + curvePhase) .* latDir;

    if classId == 2
        % UAV: 轻微高频机动
        curve = curve + 0.35 * sin(2*pi*2.0*tau + curvePhase) .* latDir;
    elseif classId == 3
        % Pedestrian: 很弱的步态摆动
        curve = curve + 0.12 * sin(2*pi*5.0*tau + curvePhase) .* latDir;
    elseif classId == 4
        % Fast maneuver: 更明显的机动弯曲
        curve = curve + 0.65 * sin(2*pi*1.2*tau + curvePhase) .* latDir;
    end

    pos = (1 - smoothTau) .* p0 + smoothTau .* p1 + curve;
    vel = estimateVelocity(pos, p.dt);

    obj.pos = pos;
    obj.vel = vel;

    objs(end + 1) = obj;
end

end


function [p0, p1] = sampleSafeMotionSegment(p)
% 在安全区域中采样一段轨迹中心线。
% 这样可以显著提高“48 帧全程可见”的成功率。

for trial = 1:300
    r0 = randUniform(22, 45);
    az0 = deg2rad(randUniform(-25, 25));
    p0 = [r0 * cos(az0), r0 * sin(az0)];

    moveLen = randUniform(22, 45);
    moveAngle = deg2rad(randUniform(-18, 18));
    dir = [cos(moveAngle), sin(moveAngle)];

    p1 = p0 + moveLen * dir;

    if isInsideSafeRegion(p, p0) && isInsideSafeRegion(p, p1)
        return;
    end
end

% fallback
p0 = [25, 0];
p1 = [65, 0];

end


function objs = generateInterferers(p, interfClassIds, startId)

n = numel(interfClassIds);
T = p.Nframes;
time = (0:T-1) * p.dt;

objs = emptyObjectArray();

for j = 1:n
    interfId = interfClassIds(j);
    cls = interferenceClassParam(interfId);

    obj.id = startId + j - 1;
    obj.isTarget = false;
    obj.targetClassId = 0;
    obj.interferenceClassId = interfId;
    obj.name = cls.name;
    obj.rcsDb = cls.rcsDb;
    obj.phase0 = 2 * pi * rand();

    r0 = randUniform(20, 85);
    az0 = deg2rad(randUniform(-55, 55));
    p0 = [r0 * cos(az0), r0 * sin(az0)];

    pos = zeros(T, 2);
    vel = zeros(T, 2);

    if interfId == 1
        % I1: 假目标 / 杂波反射体
        drift = randUniform(0.0, 0.7) * normalizeVec(randn(1, 2));

        for f = 1:T
            t = time(f);
            jitter = 0.25 * randn(1, 2);
            pos(f, :) = p0 + drift * t + jitter;
        end

        vel = estimateVelocity(pos, p.dt);

    else
        % I2: 宽带干扰源
        drift = randUniform(0.0, 1.0) * normalizeVec(randn(1, 2));

        for f = 1:T
            t = time(f);
            pos(f, :) = p0 + drift * t;
            vel(f, :) = drift;
        end
    end

    obj.pos = pos;
    obj.vel = vel;

    objs(end + 1) = obj;
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

        if obj.isTarget
            scats = targetScatterers(obj.targetClassId, f, p, obj.phase0);
            frame = addPointScatterers(frame, R, theta, radialVel, obj.rcsDb, scats, p, tFast, tSlow, rxIdx);
        else
            if obj.interferenceClassId == 1
                scats = falseTargetScatterers(obj.phase0);
                radialVel = radialVel + randn() * 1.5;
                frame = addPointScatterers(frame, R, theta, radialVel, obj.rcsDb, scats, p, tFast, tSlow, rxIdx);
            else
                frame = addWidebandJammer(frame, p, tFast, tSlow);
            end
        end
    end

    frame = addThermalNoise(frame, p);
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


function frame = addWidebandJammer(frame, p, tFast, tSlow)

amp = p.jammerScale * randUniform(0.8, 1.4);

for rx = 1:p.Nrx
    rxPhase = 2*pi*rand();

    for m = 1:p.Nchirp
        f0 = randUniform(-0.35*p.Fs, 0.35*p.Fs);
        sweepRate = randUniform(-0.25*p.Fs, 0.25*p.Fs) / max(tFast);
        phase = 2*pi*rand();

        jammer = amp * exp(1i * ( ...
            2*pi*(f0*tFast + 0.5*sweepRate*tFast.^2) + ...
            rxPhase + phase + 0.2*sin(2*pi*300*tSlow(m)) ...
        ));

        frame(:, m, rx) = frame(:, m, rx) + jammer;
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
gt.isTarget = false(nObj, 1);
gt.targetClassId = zeros(nObj, 1);
gt.interferenceClassId = zeros(nObj, 1);
gt.name = cell(nObj, 1);

gt.pos = zeros(nObj, p.Nframes, 2, 'single');
gt.vel = zeros(nObj, p.Nframes, 2, 'single');
gt.range = zeros(nObj, p.Nframes, 'single');
gt.azimuth = zeros(nObj, p.Nframes, 'single');
gt.radialVel = zeros(nObj, p.Nframes, 'single');
gt.visible = false(nObj, p.Nframes);

for k = 1:nObj
    obj = objects(k);

    gt.objId(k) = obj.id;
    gt.isTarget(k) = obj.isTarget;
    gt.targetClassId(k) = obj.targetClassId;
    gt.interferenceClassId(k) = obj.interferenceClassId;
    gt.name{k} = obj.name;

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
        gt.visible(k, f) = isVisible(p, pos);
    end
end

gt.targetObjectIndices = find(gt.isTarget);
gt.interferenceObjectIndices = find(~gt.isTarget);

end


function scats = targetScatterers(classId, frameIdx, p, phase0)
% 类别相关散射中心。
% 不同目标在 IQ / RD 图中具有不同特征。

t = (frameIdx - 1) * p.dt;

switch classId
    case 1
        % T1: 慢速平滑目标，多散射中心，稳定
        scats.weight = [1.00, 0.65, 0.45];
        scats.dr = [0.0, 1.2, -1.1];
        scats.fdOffsetHz = [0, 0, 0];
        scats.phase = phase0 + [0, 1.7, 3.1];

    case 2
        % T2: UAV，转子微多普勒边带
        rotor = 140 + 20 * sin(2*pi*0.2*t);
        scats.weight = [1.00, 0.35, 0.35, 0.20, 0.20];
        scats.dr = [0, 0.05, -0.05, 0.1, -0.1];
        scats.fdOffsetHz = [0, rotor, -rotor, 2*rotor, -2*rotor];
        scats.phase = phase0 + [0, 0.8, 1.4, 2.1, 2.8];

    case 3
        % T3: 行人，步态微多普勒
        gait = 45 + 10 * sin(2*pi*1.3*t);
        wLeg = 0.30 + 0.08 * sin(2*pi*1.3*t);
        scats.weight = [1.00, wLeg, wLeg];
        scats.dr = [0, 0.25, -0.25];
        scats.fdOffsetHz = [0, gait, -gait];
        scats.phase = phase0 + [0, 1.1, 2.2];

    case 4
        % T4: 快速机动目标
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


function scats = falseTargetScatterers(phase0)
% 假目标 / 杂波反射体：相干但不稳定

scats.weight = [1.0, 0.35];
scats.dr = [0, randUniform(-0.5, 0.5)];
scats.fdOffsetHz = [randn()*15, randn()*80];
scats.phase = phase0 + 2*pi*rand(1, 2);

end


function cls = targetClassParam(classId)

switch classId
    case 1
        cls.name = 'T1_slow_smooth';
        cls.rcsDb = 12;
        cls.curveAmp = 0.25;

    case 2
        cls.name = 'T2_uav_agile';
        cls.rcsDb = -2;
        cls.curveAmp = 0.70;

    case 3
        cls.name = 'T3_pedestrian';
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


function cls = interferenceClassParam(interfId)

switch interfId
    case 1
        cls.name = 'I1_false_reflector';
        cls.rcsDb = 18;

    case 2
        cls.name = 'I2_wideband_jammer';
        cls.rcsDb = 25;

    otherwise
        error('Unknown interference class id.');
end

end


function flag = checkTargetsVisibleAllFrames(p, targetObjs)
% 检查所有真实目标是否 48 帧内全程可见

flag = true;

for k = 1:numel(targetObjs)
    obj = targetObjs(k);

    if ~obj.isTarget
        continue;
    end

    for f = 1:p.Nframes
        pos = obj.pos(f, :);

        if ~isVisible(p, pos)
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

flag = ...
    (x > 0) && ...
    (R >= p.rMin) && ...
    (R <= p.rMax) && ...
    (abs(az) <= p.maxAbsAz);

end


function flag = isInsideSafeRegion(p, pos)

x = pos(1);
y = pos(2);

R = sqrt(x^2 + y^2);
az = atan2(y, x);

flag = ...
    (x > 0) && ...
    (R >= p.safeRMin) && ...
    (R <= p.safeRMax) && ...
    (abs(az) <= p.safeAbsAz);

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


function visualizeScene(sceneFile)

load(sceneFile, 'rdMap', 'gt', 'p', 'meta');

figure('Name', 'Example Scene Trajectories', 'Color', 'w');
hold on;
grid on;
axis equal;

drawFOV(p);

for k = 1:gt.numObjects
    xy = squeeze(gt.pos(k, :, :));

    if gt.isTarget(k)
        plot(xy(:, 1), xy(:, 2), '-o', ...
            'LineWidth', 1.5, ...
            'MarkerSize', 3);

        text(xy(1, 1), xy(1, 2), sprintf('  %s', gt.name{k}), ...
            'FontSize', 9, ...
            'Interpreter', 'none');
    else
        plot(xy(:, 1), xy(:, 2), '--x', ...
            'LineWidth', 1.2, ...
            'MarkerSize', 4);

        text(xy(1, 1), xy(1, 2), sprintf('  %s', gt.name{k}), ...
            'FontSize', 9, ...
            'Interpreter', 'none');
    end

    plot(xy(1, 1), xy(1, 2), 'go', ...
        'MarkerFaceColor', 'g', ...
        'MarkerSize', 7);

    plot(xy(end, 1), xy(end, 2), 'ks', ...
        'MarkerFaceColor', 'k', ...
        'MarkerSize', 7);
end

xlabel('x / m');
ylabel('y / m');

title(sprintf('Scene %d Trajectories | Relation = %s', ...
    meta.sceneId, meta.relationName), ...
    'Interpreter', 'none');

xlim([0, p.rMax + 5]);
ylim([-p.rMax * sind(p.fovDeg/2) - 5, ...
       p.rMax * sind(p.fovDeg/2) + 5]);

figure('Name', 'Example Range-Doppler Map', 'Color', 'w');

frameId = 1;
imagesc(p.velocityAxis, p.rangeAxis, rdMap(:, :, frameId));
axis xy;
xlabel('Radial velocity / m/s');
ylabel('Range / m');
title(sprintf('Range-Doppler Map | Scene %d | Frame %d', meta.sceneId, frameId));
colorbar;

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

plot(0, 0, 'rp', ...
    'MarkerSize', 12, ...
    'MarkerFaceColor', 'r');

text(0, 0, ' Radar', 'FontSize', 9);

end


function objs = emptyObjectArray()

template.id = [];
template.isTarget = [];
template.targetClassId = [];
template.interferenceClassId = [];
template.name = '';
template.rcsDb = [];
template.phase0 = [];
template.pos = [];
template.vel = [];

objs = repmat(template, 0, 1);

end


function y = randUniform(a, b)

y = a + (b - a) * rand();

end


function v = normalizeVec(v)

n = norm(v);

if n < 1e-8
    v = [1, 0];
else
    v = v / n;
end

end


function w = localHann(N)
% 避免依赖 Signal Processing Toolbox

n = (0:N-1)';
w = 0.5 - 0.5 * cos(2*pi*n/(N-1));

end


function deleteIfExist(filePath)

if exist(filePath, 'file')
    delete(filePath);
end

end