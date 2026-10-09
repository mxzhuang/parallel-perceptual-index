function T = time_official_modules(img_path, srmetric_dir, niqe_dir, shim_dir, nrep)
% Per-module wall-clock time (M1..M8) of the OFFICIAL Ma + NIQE code on one image.
% It calls the official functions in the same order as feature_all.m / global_gsm.m /
% quality_predict.m, only split into the modules M1..M8 described in README.md, and checks that the
% features equal feature_all's output.
%
% For a single-thread profile in MATLAB run maxNumCompThreads(1) first.
% In Octave pass shim_dir (see setup_octave.sh); in MATLAB pass ''.
%
% Example:
%   T = time_official_modules('tests/images/coffee_y.png', 'sr-metric', 'niqe_release', '', 3)

if nargin < 5, nrep = 1; end
if exist('OCTAVE_VERSION', 'builtin')
  pkg load image
  pkg load signal
end
addpath(niqe_dir);
addpath(fullfile(srmetric_dir, 'external', 'randomforest-matlab', 'RF_Reg_C'));
addpath(fullfile(srmetric_dir, 'external', 'matlabPyrTools'));
addpath(srmetric_dir);
if ~isempty(shim_dir), addpath(shim_dir); end

S = load(fullfile(srmetric_dir, 'model.mat'));  % not timed
model = S.model;
P = load(fullfile(niqe_dir, 'modelparameters.mat'));
img = imread(img_path);
if ndims(img) == 3, img = rgb2gray(img); end

T = zeros(1, 8);
for rep = 1:nrep
  % M1 spatial pyramid
  t0 = tic;
  im1 = im2double(img);
  h = fspecial('gaussian', 3);
  im_f = double(imfilter(im1, h)); im2 = im_f(2:2:end, 2:2:end);
  im_f = double(imfilter(im2, h)); im3 = im_f(2:2:end, 2:2:end);
  T(1) = T(1) + toc(t0);

  % M2 block-DCT statistics
  t0 = tic;
  f1 = [block_dct(im1) block_dct(im2) block_dct(im3)];
  T(2) = T(2) + toc(t0);

  % M3 patch SVD
  t0 = tic;
  f3 = [svd(im2col(im1, [5 5], 'distinct')) svd(im2col(im2, [5 5], 'distinct')) svd(im2col(im3, [5 5], 'distinct'))];
  T(3) = T(3) + toc(t0);

  % M4 steerable pyramid (global_gsm.m)
  t0 = tic;
  im = double(img);
  [pyr, pind] = buildSFpyr(im, 2, 5);
  T(4) = T(4) + toc(t0);

  % M5 divisive normalisation
  t0 = tic;
  [subband, size_band] = norm_sender_normalized(pyr, pind, 2, 6, 1, 1, 3, 3, 50);
  T(5) = T(5) + toc(t0);

  % M6 subband statistics (rest of global_gsm.m)
  t0 = tic;
  f = [];
  gama_horz = zeros(1, length(subband));
  for ii = 1:length(subband), gama_horz(ii) = gama_gen_gauss(subband{ii}); end
  f = [f, gama_horz];
  gama_scale = zeros(1, length(subband) / 2);
  for ii = 1:length(subband) / 2
    gama_scale(ii) = gama_gen_gauss([subband{ii}; subband{ii + 6}]);
  end
  f = [f, gama_scale];
  hp_band = pyrBand(pyr, pind, 1);
  cs_val = zeros(1, length(subband));
  for ii = 1:length(subband)
    curr_band = pyrBand(pyr, pind, ii + 1);
    [~, ~, cs_val(ii)] = ssim_index_new(imresize(curr_band, size(hp_band)), hp_band);
  end
  f = [f, cs_val];
  cs_val = [];
  nn = 1;
  for i = 1:length(subband) / 2
    for j = i + 1:length(subband) / 2
      [~, ~, cs_val(nn)] = ssim_index_new(reshape(subband{i}, size_band(i, :)), reshape(subband{j}, size_band(j, :)));
      nn = nn + 1;
    end
  end
  f2 = [f, cs_val];
  T(6) = T(6) + toc(t0);

  % M7 NIQE
  t0 = tic;
  nq = computequality(img, 96, 96, 0, 0, P.mu_prisparam, P.cov_prisparam);
  T(7) = T(7) + toc(t0);

  % M8 regression and fusion
  t0 = tic;
  s1 = regRF_predict(reshape(f1, 1, []), model.rf{1});
  s2 = regRF_predict(reshape(f2, 1, []), model.rf{2});
  s3 = regRF_predict(reshape(f3, 1, []), model.rf{3});
  ma = [1 s1 s2 s3] * model.linear;
  T(8) = T(8) + toc(t0);
end
T = T / nrep;

% sanity check against the unmodified feature_all
[g1, g2, g3] = feature_all(img);
assert(isequal(f1, g1) && isequal(f2, g2) && isequal(f3, g3), 'features differ from feature_all');

fprintf('Ma = %.6f  NIQE = %.6f  PI = %.6f\n', ma, nq, ((10 - ma) + nq) / 2);
for m = 1:8
  fprintf('M%d  %8.3f s  %5.1f%%\n', m, T(m), 100 * T(m) / sum(T));
end
end
