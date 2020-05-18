% conditional_mvnpdf_low_rank: efficiently computes
%
% p(y1 | y2, λ, v, ω, z_qso, M) = N(y1; μ1', Σ11')
%                    [ Σ11    Σ12 ] 
%  μ = [μ1, μ2]; Σ = |            |
%                    [ Σ21    Σ22 ]
% The μ1 will be updated:
%  μ1' = μ1 + Σ12 Σ22^-1 (y2 - μ2)
% The Σ11 will be updated:
%  Σ11' = Σ11 - Σ12 Σ22^-1 Σ21
% 
% Note Σ = MM' + diag(d)


function [mu1, Sigma11] = conditional_mvnpdf_low_rank(y2, mu1, mu2, M1, M2, d1, d2)

  [n1, k1] = size(M1);
  [n2, k2] = size(M2);

  assert(k1 == k2)
  k = k1;

  % y2 is the observation we want to condition on
  y2 = y2 - (mu2);                    % (y2 - μ2)

  d2_inv = 1 ./ d2;

  % build the inverse Σ22
  D2_inv_M2 = d2_inv .* M2;
  D2_inv_y2 = d2_inv .* y2;

  % use Woodbury identity, define
  %   B = (I + M' D^-1 M),
  % then
  %   K^-1 = D^-1 - D^-1 M B^-1 M' D^-1
  B2 = M2' * D2_inv_M2;
  B2(1:(k + 1):end) = B2(1:(k + 1):end) + 1;
  
  L2 = chol(B2);
  
  % C = B^-1 M' D^-1
  % shape (k, n2)
  C2 = L2 \ (L2' \ D2_inv_M2');

  % K22_inv = d2_inv - D2_inv_M2 * (C2);

  K22_inv_y       = D2_inv_y2 - D2_inv_M2 * (C2 * y2);
  K22_inv_Sigma21 = d2_inv .* (M2 * M1') - D2_inv_M2 * (C2 * (M2 * M1'));

  % μ1' = μ1 + Σ12 Σ22^-1 (y2 - μ2)
  mu1 = mu1 + ((M1 * M2') * K22_inv_y);
  
  % Σ11' = Σ11 - Σ12 Σ22^-1 Σ21
  Sigma11 = diag(d1) + M1 * M1' - (M1 * M2') * K22_inv_Sigma21;
end
