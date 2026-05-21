function visualize_all_fmcw_scenes(datasetDir, scenesPerPage, saveEachScene, makeVideos)
% VISUALIZE_ALL_FMCW_SCENES
%
% 可视化所有 FMCW 雷达仿真场景。
%
% 当前版本适配新的数据生成代码：
%   1. 每个场景只有连续 48 帧；
%   2. 数据生成阶段不再保存 inputFrames / predFrames；
%   3. 可视化时不再强行区分前 32 帧和后 16 帧；
%   4. 直接展示完整 48 帧轨迹；
%   5. 详细图中展示第 1、24、48 帧 Range-Doppler 图。
%
% 用法：
%   visualize_all_fmcw_scenes('fmcw_traj_dataset');
%
% 或者：
%   visualize_all_fmcw_scenes('fmcw_traj_dataset', 16, true, false);
%
% 参数：
%   datasetDir     : 数据集文件夹
%   scenesPerPage  : 每页显示多少个场景，默认 16
%   saveEachScene  : 是否保存每个场景的详细图，默认 true
%   makeVideos     : 是否保存每个场景动画，默认 false
%
% 输出：
%   fmcw_traj_dataset/visualization/
%       overview_page_001.png
%       overview_page_002.png
%       ...
%       scene_000001_detail.png
%       scene_000002_detail.png
%       ...
%       videos/scene_000001_animation.mp4

if nargin < 1
    datasetDir = 'fmcw_traj_dataset';
end

if nargin < 2
    scenesPerPage = 16;
end

if nargin < 3
    saveEachScene = true;
end

if nargin < 4
    makeVideos = false;
end

indexFile = fullfile(datasetDir, 'index.csv');

if ~exist(indexFile, 'file')
    error('Cannot find index.csv in folder: %s', datasetDir);
end

indexTable = readtable(indexFile);
numScenes = height(indexTable);

visDir = fullfile(datasetDir, 'visualization');

if ~exist(visDir, 'dir')
    mkdir(visDir);
end

fprintf('Dataset folder : %s\n', datasetDir);
fprintf('Total scenes   : %d\n', numScenes);
fprintf('Scenes per page: %d\n', scenesPerPage);
fprintf('Output folder  : %s\n\n', visDir);

% 1. 所有场景分页总览
plotAllSceneOverviewPages(datasetDir, indexTable, scenesPerPage, visDir);

% 2. 每个场景保存详细图
if saveEachScene
    fprintf('\nSaving detailed figures for all scenes...\n');

    for i = 1:numScenes
        sceneFileName = getSceneFileName(indexTable, i);
        sceneFile = fullfile(datasetDir, sceneFileName);

        savePath = fullfile(visDir, sprintf('scene_%06d_detail.png', i));

        plotOneSceneDetailAndSave(sceneFile, savePath);

        fprintf('Saved detail figure: %s\n', savePath);
    end
end

% 3. 可选：每个场景保存动画
if makeVideos
    fprintf('\nSaving animation videos for all scenes...\n');

    videoDir = fullfile(visDir, 'videos');

    if ~exist(videoDir, 'dir')
        mkdir(videoDir);
    end

    for i = 1:numScenes
        sceneFileName = getSceneFileName(indexTable, i);
        sceneFile = fullfile(datasetDir, sceneFileName);

        videoPath = fullfile(videoDir, sprintf('scene_%06d_animation.mp4', i));

        saveOneSceneAnimation(sceneFile, videoPath);

        fprintf('Saved animation: %s\n', videoPath);
    end
end

fprintf('\nAll visualization finished.\n');
fprintf('Please check folder: %s\n', visDir);

end


function plotAllSceneOverviewPages(datasetDir, indexTable, scenesPerPage, visDir)
% 所有场景分页总览。
% 每个小图显示一个完整 48 帧场景。

numScenes = height(indexTable);
numPages = ceil(numScenes / scenesPerPage);

for pageId = 1:numPages
    startIdx = (pageId - 1) * scenesPerPage + 1;
    endIdx = min(pageId * scenesPerPage, numScenes);
    sceneIdxList = startIdx:endIdx;

    nThisPage = numel(sceneIdxList);
    nCol = ceil(sqrt(scenesPerPage));
    nRow = ceil(scenesPerPage / nCol);

    fig = figure( ...
        'Name', sprintf('All Scenes Overview Page %d', pageId), ...
        'Color', 'w', ...
        'Position', [100, 100, 1600, 1000]);

    tiledlayout(nRow, nCol, ...
        'Padding', 'compact', ...
        'TileSpacing', 'compact');

    for t = 1:nThisPage
        i = sceneIdxList(t);

        sceneFileName = getSceneFileName(indexTable, i);
        sceneFile = fullfile(datasetDir, sceneFileName);

        load(sceneFile, 'gt', 'p', 'meta');

        nexttile;
        hold on;
        grid on;
        axis equal;

        drawFOV(p);

        for k = 1:gt.numObjects
            xy = squeeze(gt.pos(k, :, :));
            vis = gt.visible(k, :);

            if gt.isTarget(k)
                % 真实目标：实线
                plot(xy(:, 1), xy(:, 2), '-o', ...
                    'LineWidth', 1.2, ...
                    'MarkerSize', 2);

                % 目标理论上应该全程可见
                if any(~vis)
                    plot(xy(~vis, 1), xy(~vis, 2), 'rx', ...
                        'LineWidth', 1.2, ...
                        'MarkerSize', 5);
                end
            else
                % 干扰物：虚线
                plot(xy(:, 1), xy(:, 2), '--x', ...
                    'LineWidth', 1.0, ...
                    'MarkerSize', 3);

                % 对干扰物，额外标记可见点
                plot(xy(vis, 1), xy(vis, 2), '.', ...
                    'MarkerSize', 8);
            end

            % 起点和终点
            plot(xy(1, 1), xy(1, 2), 'go', ...
                'MarkerFaceColor', 'g', ...
                'MarkerSize', 4);

            plot(xy(end, 1), xy(end, 2), 'ks', ...
                'MarkerFaceColor', 'k', ...
                'MarkerSize', 4);
        end

        xlabel('x / m');
        ylabel('y / m');

        title(sprintf('Scene %d | T=%d, I=%d | %s', ...
            meta.sceneId, meta.nTargets, meta.nInterferers, meta.relationName), ...
            'Interpreter', 'none', ...
            'FontSize', 8);

        xlim([0, p.rMax + 5]);
        ylim([-p.rMax * sind(p.fovDeg / 2) - 5, ...
               p.rMax * sind(p.fovDeg / 2) + 5]);
    end

    sgtitle(sprintf('All Scene Overview | Page %d / %d', pageId, numPages));

    savePath = fullfile(visDir, sprintf('overview_page_%03d.png', pageId));
    saveFigure(fig, savePath);

    fprintf('Saved overview page: %s\n', savePath);
end

end


function plotOneSceneDetailAndSave(sceneFile, savePath)
% 单个场景详细图：
% 左边：完整 48 帧二维轨迹；
% 右边：第 1、24、48 帧 Range-Doppler 图。

load(sceneFile, 'gt', 'p', 'meta', 'rdMap');

fig = figure( ...
    'Visible', 'off', ...
    'Color', 'w', ...
    'Position', [100, 100, 1700, 900]);

tiledlayout(2, 3, ...
    'Padding', 'compact', ...
    'TileSpacing', 'compact');

% ==============================
% 左侧：完整 48 帧轨迹
% ==============================
nexttile([2, 1]);
hold on;
grid on;
axis equal;

drawFOV(p);

for k = 1:gt.numObjects
    xy = squeeze(gt.pos(k, :, :));
    vis = gt.visible(k, :);

    if gt.isTarget(k)
        plot(xy(:, 1), xy(:, 2), '-o', ...
            'LineWidth', 1.8, ...
            'MarkerSize', 4);

        text(xy(1, 1), xy(1, 2), ...
            sprintf('  Target-%d: %s', k, gt.name{k}), ...
            'FontSize', 8, ...
            'Interpreter', 'none');

        % 如果目标有不可见帧，用红叉标记，方便检查数据是否异常
        if any(~vis)
            plot(xy(~vis, 1), xy(~vis, 2), 'rx', ...
                'LineWidth', 2.0, ...
                'MarkerSize', 8);
        end

    else
        plot(xy(:, 1), xy(:, 2), '--x', ...
            'LineWidth', 1.5, ...
            'MarkerSize', 5);

        text(xy(1, 1), xy(1, 2), ...
            sprintf('  Interf-%d: %s', k, gt.name{k}), ...
            'FontSize', 8, ...
            'Interpreter', 'none');

        % 干扰物可能并非全程可见，这里把可见点额外标出来
        plot(xy(vis, 1), xy(vis, 2), '.', ...
            'MarkerSize', 10);
    end

    % 起点
    plot(xy(1, 1), xy(1, 2), 'go', ...
        'MarkerFaceColor', 'g', ...
        'MarkerSize', 7);

    % 终点
    plot(xy(end, 1), xy(end, 2), 'ks', ...
        'MarkerFaceColor', 'k', ...
        'MarkerSize', 7);

    % 每隔 8 帧标一个帧号，方便观察运动方向
    frameMarks = 1:8:p.Nframes;
    for m = frameMarks
        text(xy(m, 1), xy(m, 2), sprintf(' %d', m), ...
            'FontSize', 7, ...
            'Color', [0.2, 0.2, 0.2]);
    end
end

xlabel('x / m');
ylabel('y / m');

title(sprintf('Scene %d Full 48-frame Trajectory | Relation = %s', ...
    meta.sceneId, meta.relationName), ...
    'Interpreter', 'none');

xlim([0, p.rMax + 5]);
ylim([-p.rMax * sind(p.fovDeg / 2) - 5, ...
       p.rMax * sind(p.fovDeg / 2) + 5]);

% ==============================
% 右侧：Range-Doppler 图
% ==============================
frameList = unique(round([1, p.Nframes / 2, p.Nframes]));
frameNameList = cell(numel(frameList), 1);

for i = 1:numel(frameList)
    frameNameList{i} = sprintf('Frame %d', frameList(i));
end

for i = 1:numel(frameList)
    nexttile;

    frameId = frameList(i);

    imagesc(p.velocityAxis, p.rangeAxis, rdMap(:, :, frameId));
    axis xy;

    xlabel('Radial velocity / m/s');
    ylabel('Range / m');

    title(sprintf('Range-Doppler | %s', frameNameList{i}), ...
        'Interpreter', 'none');

    colorbar;
end

sgtitle(sprintf('Scene %d Detail | Targets = %d | Interferers = %d | Frames = %d', ...
    meta.sceneId, meta.nTargets, meta.nInterferers, p.Nframes), ...
    'Interpreter', 'none');

saveFigure(fig, savePath);
close(fig);

end


function saveOneSceneAnimation(sceneFile, videoPath)
% 保存单个场景动画：
% 左边：二维轨迹随时间变化；
% 右边：当前帧 Range-Doppler 图。

load(sceneFile, 'gt', 'p', 'meta', 'rdMap');

fig = figure( ...
    'Visible', 'off', ...
    'Color', 'w', ...
    'Position', [100, 100, 1400, 600]);

vw = VideoWriter(videoPath, 'MPEG-4');
vw.FrameRate = p.frameRate;
open(vw);

for f = 1:p.Nframes
    clf(fig);

    tiledlayout(1, 2, ...
        'Padding', 'compact', ...
        'TileSpacing', 'compact');

    % ==============================
    % 左侧：轨迹动画
    % ==============================
    nexttile;
    hold on;
    grid on;
    axis equal;

    drawFOV(p);

    for k = 1:gt.numObjects
        xy = squeeze(gt.pos(k, :, :));
        vis = gt.visible(k, :);

        past = xy(1:f, :);
        current = xy(f, :);

        if gt.isTarget(k)
            plot(past(:, 1), past(:, 2), '-o', ...
                'LineWidth', 1.8, ...
                'MarkerSize', 3);

            plot(current(1), current(2), 'ro', ...
                'MarkerFaceColor', 'r', ...
                'MarkerSize', 8);

            text(current(1), current(2), ...
                sprintf('  %s', gt.name{k}), ...
                'FontSize', 8, ...
                'Interpreter', 'none');

            if ~vis(f)
                plot(current(1), current(2), 'rx', ...
                    'LineWidth', 2.0, ...
                    'MarkerSize', 10);
            end

        else
            plot(past(:, 1), past(:, 2), '--x', ...
                'LineWidth', 1.4, ...
                'MarkerSize', 4);

            if vis(f)
                plot(current(1), current(2), 'ms', ...
                    'MarkerFaceColor', 'm', ...
                    'MarkerSize', 7);
            else
                plot(current(1), current(2), 'kx', ...
                    'LineWidth', 1.5, ...
                    'MarkerSize', 7);
            end

            text(current(1), current(2), ...
                sprintf('  %s', gt.name{k}), ...
                'FontSize', 8, ...
                'Interpreter', 'none');
        end
    end

    xlabel('x / m');
    ylabel('y / m');

    title(sprintf('Scene %d | Frame %d/%d', ...
        meta.sceneId, f, p.Nframes), ...
        'Interpreter', 'none');

    xlim([0, p.rMax + 5]);
    ylim([-p.rMax * sind(p.fovDeg / 2) - 5, ...
           p.rMax * sind(p.fovDeg / 2) + 5]);

    % ==============================
    % 右侧：当前 RD 图
    % ==============================
    nexttile;

    imagesc(p.velocityAxis, p.rangeAxis, rdMap(:, :, f));
    axis xy;

    xlabel('Radial velocity / m/s');
    ylabel('Range / m');

    title(sprintf('Range-Doppler Map | Frame %d', f));

    colorbar;

    drawnow;

    frame = getframe(fig);
    writeVideo(vw, frame);
end

close(vw);
close(fig);

end


function drawFOV(p)
% 画雷达视场

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
    'MarkerSize', 10, ...
    'MarkerFaceColor', 'r');

text(0, 0, ' Radar', 'FontSize', 8);

end


function saveFigure(fig, savePath)
% 保存图片，兼容不同 MATLAB 版本

try
    exportgraphics(fig, savePath, 'Resolution', 200);
catch
    saveas(fig, savePath);
end

end


function sceneFileName = getSceneFileName(indexTable, i)
% 兼容 readtable 读入 cell/string/categorical 的情况

fileCol = indexTable.file;

if iscell(fileCol)
    sceneFileName = fileCol{i};
elseif isstring(fileCol)
    sceneFileName = fileCol(i);
    sceneFileName = char(sceneFileName);
elseif iscategorical(fileCol)
    sceneFileName = char(fileCol(i));
else
    sceneFileName = char(fileCol(i));
end

end