%%-------------------------------����1��ˮ�棬�ٶȱ仯-------------------------------
%%%�趨�ĳ�����
format long
Point.lat = [35,   34.5, 34, 33, 32];%γ��
Point.lon = [120,  120.5,  120.8, 121, 122.5];%����
Point.height = [0, 0, 0, 0,0];%�߶�
Point.speed = [10, 20, 10, 20,10];
Point.changeDirection = [0,-1,-1, 1,0];%�Ƿ�ת���־
Point.n = [2, 2, 2,2,2];%ת���������
Point.W = [10, 10, 10,10,10];%�ı���ٵĲ�����������/��
Point.a = [0.05,0.05, 0.05,0.05,0.05];%�ı��ٶȵĲ�����������/��^2
delta_T = 1;
% az1 = azimuth(Point.lat(1),Point.lon(1),Point.lat(2),Point.lon(2),referenceEllipsoid('wgs84'));%���ص��ǵ�ֵ
% az2 = azimuth(Point.lat(2),Point.lon(2),Point.lat(3),Point.lon(3),referenceEllipsoid('wgs84'));
%��������
pos = GJJS_MainFun( Point,delta_T,0);
%���������ݸ�ʽת��
n=length(pos);%�����·����ĸ���
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
    posSeq(i,12)=pos(i).K; %·����ĺ���ǣ��ȡ�
end
%����ͼ����
figure
worldmap([31.5 35.5],[119.5 123]);
geoshow('landareas.shp', 'FaceColor', [0.5 1.0 0.5]);
hold on;
plotm(Point.lat,Point.lon,'c*-');%���趨�Ĺ켣ͼ
plotm(posSeq(:,2),posSeq(:,3),'b-');%%%�������������ɵĹ켣
hold off
% %������ͼ
% figure
% plot(posSeq(1:4500,5),'.b')
% title('ˮ������ٶȱ仯')
% ylabel('�ٶȣ�m/s��')
% xlabel('�켣������(ÿ��)')
%д���ļ�
tIndex=1:60:n;
valueIndex=[1,2,3,4,5,12];
% csvwrite('����1��ˮ������ٶȻ�����ÿ�룩.csv',posSeq(:,valueIndex));%�зֱ�Ϊ��ʱ�䣨ÿ�룩��γ�ȣ����ȣ��߶ȣ�m��,�ٶȣ�m/s��,����
dlmwrite('����1��ˮ������ٶȻ�����ÿ�룩.csv',posSeq(:,valueIndex),'precision', '%.10f');%�зֱ�Ϊ��ʱ�䣨ÿ�룩��γ�ȣ����ȣ��߶ȣ�m��,�ٶȣ�m/s��,����