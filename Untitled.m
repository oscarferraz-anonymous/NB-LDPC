rem(182,15)

H=zeros(8,16);H(1,1)=182;H(1,5)=8;H(1,9)=173;
H(1,16)=1;H(2,2)=89;H(2,6)=1;H(2,10)=9;H(2,13)=81;
H(3,3)=1;H(3,7)=173;H(3,11)=182;H(3,14)=8;
H(4,4)=8;H(4,8)=182;H(4,12)=173;H(4,15)=1;
H(5,3)=88;H(5,8)=80;H(5,9)=1;H(5,13)=8;
H(6,4)=169;H(6,5)=1;H(6,10)=127;H(6,14)=40;
H(7,1)=169;H(7,6)=128;H(7,11)=40;H(7,15)=1;
H(8,2)=8;H(8,7)=80;H(8,12)=88;H(8,16)=1;

b=[0
24
4
28
8
16
12
20
1
21
5
25
9
29
13
17
2
18
6
22
10
26
14
30
7
19
11
23
15
27
3
31     ];
b=b*2

aux=[]

for i=1:32
    aux(i,:)= (find(new(i,:))-1)
end



