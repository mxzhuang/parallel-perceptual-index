function run_reference(img_dir, out_file, srmetric_dir, niqe_dir, shim_dir)
% Run the OFFICIAL Ma (sr-metric) and NIQE code on every *_y.png in img_dir and
% write all intermediate features and scores to out_file (one line per image).
%
% Works in MATLAB and in GNU Octave. In Octave, shim_dir must contain the
% MATLAB-compatible replacements (imresize.m, blkproc.m, nanmean.m, nancov.m) and
% the Octave builds of pointOp / mexRF_predict; it is put first on the path.
%
% Images must already be in PIRM form: Y channel of rgb2ycbcr, 4-pixel shave
% (tools/make_test_images.py produces them).
%
% Output line format (space separated, %.17g):
%   name  ma  niqe  s1 s2 s3  n_f1 f1...  n_f2 f2...  n_f3 f3...

if exist('OCTAVE_VERSION', 'builtin')
  pkg load image
  pkg load signal
end
addpath(niqe_dir);
addpath(fullfile(srmetric_dir, 'external', 'randomforest-matlab', 'RF_Reg_C'));
addpath(fullfile(srmetric_dir, 'external', 'matlabPyrTools'));
addpath(srmetric_dir);
if nargin >= 5 && ~isempty(shim_dir)
  addpath(shim_dir);  % last addpath = first on the path
end

S = load(fullfile(srmetric_dir, 'model.mat'));
model = S.model;
P = load(fullfile(niqe_dir, 'modelparameters.mat'));

files = dir(fullfile(img_dir, '*_y.png'));
fid = fopen(out_file, 'w');
for k = 1:numel(files)
  name = files(k).name;
  im = imread(fullfile(img_dir, name));
  t0 = tic;
  [f1, f2, f3] = feature_all(im);
  s1 = regRF_predict(reshape(f1, 1, []), model.rf{1});
  s2 = regRF_predict(reshape(f2, 1, []), model.rf{2});
  s3 = regRF_predict(reshape(f3, 1, []), model.rf{3});
  ma = [1 s1 s2 s3] * model.linear;
  t_ma = toc(t0);
  t0 = tic;
  nq = computequality(im, 96, 96, 0, 0, P.mu_prisparam, P.cov_prisparam);
  t_nq = toc(t0);
  fprintf(fid, '%s %.17g %.17g %.17g %.17g %.17g', name, ma, nq, s1, s2, s3);
  for f = {f1, f2, f3}
    v = f{1}(:);
    fprintf(fid, ' %d', numel(v));
    fprintf(fid, ' %.17g', v);
  end
  fprintf(fid, '\n');
  fprintf('%s  Ma=%.6f  NIQE=%.6f  PI=%.6f  (Ma %.1fs, NIQE %.1fs)\n', ...
          name, ma, nq, ((10 - ma) + nq) / 2, t_ma, t_nq);
end
fclose(fid);
end
