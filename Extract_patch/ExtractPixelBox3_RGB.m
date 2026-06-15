function [ListU] = ExtractPixelBox3_RGB(i1, j1, w, l, RotatedImage, im, RotMatfull)
    % Get image dimensions
    [oldY, oldX, numChannels] = size(im);
    [newY, newX, ~] = size(RotatedImage);
    
    % Initialize list
    if numChannels == 3
        list = zeros(round(l), round(w), 3);
    else
        list = zeros(round(l), round(w));
    end
    
    point = zeros(l*w, 3);
    start = floor([i1, j1]);
    count = 1;
    
    % Check if the cropping box exceeds the boundaries of the rotated image
    if start(1) < 1 || start(1) + l - 1 > newY || ...
       start(2) < 1 || start(2) + w - 1 > newX
        % Return empty array if out of bounds
        ListU = [];
        return;
    end
    
    % Generate coordinates before rotation
    for i = 1:l
        for j = 1:w
            point(count, :) = [i+start(1)-1, j+start(2)-1, 1];     
            count = count + 1;
        end
    end    
    
    % Compute coordinates after rotation
    pointRo = floor(RotMatfull \ point');
    
    % Check whether all rotated points lie within the original image boundaries
    allPointsValid = true;
    for idx = 1:size(pointRo, 2)
        pointR = pointRo(:, idx);
        if pointR(1) < 1 || pointR(1) > oldY || pointR(2) < 1 || pointR(2) > oldX
            allPointsValid = false;
            break;
        end
    end
    
    % Return empty array if any point is out of bounds of the original image
    if ~allPointsValid
        ListU = [];
        return;
    end
    
    % All points are within bounds, assign pixel values
    count = 1;
    for i = 1:l
        for j = 1:w
            pointR = pointRo(:, count);
            
            % Handle multi-channel images
            if numChannels == 3
                list(i, j, 1) = im(pointR(1), pointR(2), 1);  % R channel
                list(i, j, 2) = im(pointR(1), pointR(2), 2);  % G channel
                list(i, j, 3) = im(pointR(1), pointR(2), 3);  % B channel
            else
                list(i, j) = im(pointR(1), pointR(2));        % Single channel
            end
            count = count + 1;
        end
    end
    
    % Convert to appropriate type
    if numChannels == 3
        ListU = uint8(list);  % RGB image
    else
        ListU = uint8(list);  % Grayscale image
    end
end