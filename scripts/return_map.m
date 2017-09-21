function map = return_map(B_train, B_test, S)
    [distH, orderH] = calcHammingRank (B_train, B_test) ;
    %save('data2.mat','B_train','B_test','S','distH','orderH');
    map = calcMAP(orderH,S');
end