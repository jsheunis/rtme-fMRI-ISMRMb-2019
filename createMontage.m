function montage = createMontage(img, columns, rotate, str, clrmp, mask, CLim)

% stephan edit 5 october: this is the main version of the createMontage
% function that i use. need to consolidate this with other versions.

% stephan edit 2 october: in plane dimensions do not match (i.e. Ni != Nj)
% which means there are dimension mismatch errors after rotation, need to
% fix this. tried some stuff, need to do a complete overhaul of the code at
% some point.

montage = struct;

[Ni, Nj, Nk] = size(img);

if rotate
    new_img = zeros(Nj, Ni, Nk);
    for p = 1:Nk
        new_img(:,:,p) = rot90(img(:,:,p));
    end
    img = new_img;
    filler = zeros(Nj, Ni);
else
    filler = zeros(Ni, Nj);
end



rows = floor(Nk/columns);
fill = mod(Nk, columns);
if fill == 0
    N_fill = 0;
else
    N_fill = columns - mod(Nk, columns);
end

montage.rows = rows;
montage.columns = columns;
montage.N_fill = N_fill;

assignin('base', 'montagexxx', montage)

parts = {};

for i = 1:rows
    for j = 1:columns
        if j ==1
            parts{i} = img(:,:,(columns*(i-1)+j));
        else
            parts{i} = cat(2, parts{i}, img(:,:,(columns*(i-1)+j)));
        end
    end
    if i ==1
        whole = parts{i};
    else
        whole = cat(1, whole, parts{i});
    end
end

if N_fill ~= 0
    % last row
    last_parts = img(:,:,(rows*columns+1));
    for k = (rows*columns+2):Nk
        last_parts = cat(2, last_parts, img(:,:,k));
    end
    for m = 1:N_fill
        last_parts = cat(2, last_parts, filler);
    end
    montage.image = cat(1, whole, last_parts);
else
    montage.image = whole;
end



f = figure; imagesc(montage.image); colormap(clrmp); colorbar;ax = gca; ax.CLim = CLim;
title(str);
montage.f = f;





