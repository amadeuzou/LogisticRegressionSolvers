function lambda = lrLinearSearch(func, param, lambda0, step0)
% http://en.wikipedia.org/wiki/Line_search
% get search range
[a, b] = getSearchRange(func, param, lambda0, step0);
l = a + 0.382*(b-a);
u = a + 0.618*(b-a);
% linear search
k=1;
tol = b-a;
max_itr = 1000;
eps = 1e-3;
while tol>eps && k<max_itr
    fl = func(param, l);
    fu = func(param, u);
    if fl > fu
        a = l;
        l = u;
        u = a + 0.618*(b - a);
    else
        b = u;
        u = l;
        l = a + 0.382*(b-a);
    end
    k = k+1;
    tol = abs(b - a);
end
if k == max_itr
    disp('get max iterator');
    x0 = lambda0;
    return;
end
x0 = (a+b)/2;
lambda = x0;

function [minx,maxx] = getSearchRange(func, param, x0, h0)

x1 = x0;
k = 0;
h = h0;
while 1
    x4 = x1 + h;
    k = k+1;
    f4 = func(param, x4);
    f1 = func(param, x1);
    if f4 < f1
        x2 = x1;
        x1 = x4;
        f2 = f1;
        f1 = f4;
        h = 2*h;
    else
        if k==1
            h = -h;
            x2 = x4;
            f2 = f4;
        else
            x3 = x2;
            x2 = x1;
            x1 = x4;
            break;
        end
    end
end

minx = min(x1,x3);
maxx = x1+x3 - minx;
