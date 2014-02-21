function I = yacupad(I, varargin)

if nargin < 2
    initsize = size(I);
else
    initsize = max(size(I), varargin{1});
end

n = [2 3 5 7];
padsize = arrayfun(@(a) lgm(a, n), initsize);
I = padarray(padarray(I, floor((padsize - size(I))/2), 'circular'), mod(padsize - size(I), 2), 'circular', 'pre');

end

function r = lgm(a, n) % least greater multiple
r = a;
while(~ismultiple(r, n))
    r = r+1;
end
end

function r = ismultiple(a, n)
while a > 1
    m = (mod(a, n) == 0);
    if ~any(m)
        r = false;
        return
    end
    b = n(m);
    a = a / b(1);
end
r = true;
end

