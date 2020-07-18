%%-------------------------------用例1：水面，速度变化-------------------------------
%%%设定的出发点
format long
Point.lat = [35,   34.5, 34, 33, 32];%纬度
Point.lon = [120,  120.5,  120.8, 121, 122.5];%经度
Point.height = [0, 0, 0, 0,0];%高度
Point.speed = [10, 20, 10, 20,10];
Point.changeDirection = [0,-1,-1, 1,0];%是否转向标志
Point.n = [2, 2, 2,2,2];%转弯参数向量
Point.W = [10, 10, 10,10,10];%改变高速的参数向量，米/秒
Point.a = [0.05,0.05, 0.05,0.05,0.05];%改变速度的参数向量，米/秒^2
delta_T = 1;
% az1 = azimuth(Point.lat(1),Point.lon(1),Point.lat(2),Point.lon(2),referenceEllipsoid('wgs84'));%返回的是单值
% az2 = azimuth(Point.lat(2),Point.lon(2),Point.lat(3),Point.lon(3),referenceEllipsoid('wgs84'));
%兵力解算
pos = GJJS_MainFun( Point,delta_T,0);
%计算结果数据格式转化
n=length(pos);%补充的路径点的个数
posSeq=zeros(n,12);
for i=1:n
    posSeq(i,1)=i;
    posSeq(i,2)=pos(i).lat;
    posSeq(i,3)=pos(i).lon;
    posSeq(i,4)=pos(i).height;
    posSeq(i,5)=pos(i).speed;
    posSeq(i,6)=pos(i).changeDirection;
    posSeq(i,7)=pos(i).changeHeight;
    posSeq(i,8)=pos(i).changeSpeed;
    posSeq(i,9)=pos(i).n;
    posSeq(i,10)=pos(i).W;
    posSeq(i,11)=pos(i).a;
    posSeq(i,12)=pos(i).K; %路径点的航向角，度。
end
%画海图背景
figure
worldmap([31.5 35.5],[119.5 123]);
geoshow('landareas.shp', 'FaceColor', [0.5 1.0 0.5]);
hold on;
plotm(Point.lat,Point.lon,'c*-');%画设定的轨迹图
plotm(posSeq(:,2),posSeq(:,3),'b-');%%%画兵力解算生成的轨迹
hold off
% %画折线图
% figure
% plot(posSeq(1:4500,5),'.b')
% title('水面兵力速度变化')
% ylabel('速度（m/s）')
% xlabel('轨迹点序列(每秒)')
%写入文件
tIndex=1:60:n;
valueIndex=[1,2,3,4,5,12];
% csvwrite('用例1：水面兵力速度机动（每秒）.csv',posSeq(:,valueIndex));%列分别为：时间（每秒），纬度，经度，高度（m）,速度（m/s）,航向
dlmwrite('用例1：水面兵力速度机动（每秒）.csv',posSeq(:,valueIndex),'precision', '%.10f');%列分别为：时间（每秒），纬度，经度，高度（m）,速度（m/s）,航向