function I = yacupad(I)

n = [2 3 5 7];
padsize = arrayfun(@(a) lgm(a, n), size(I));
I = padarray(padarray(I, floor((padsize - size(I))/2)), mod(padsize - size(I), 2), 'pre');

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

