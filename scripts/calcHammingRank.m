function [distH, orderH] = calcHammingRank (B1, B2)
  distH = calcHammingDist(B2, B1);
  [~, orderH] = sort(distH,2);
  %distH(end,1:10)
end
