function [imCroped] = imCropFromMidpoint(im, Line, boxWidth, boxHeight)
    x1 = Line(1); y1 = Line(2);
    x2 = Line(3); y2 = Line(4);
    
    % Compute the midpoint of the line segment
    midX = (x1 + x2) / 2;
    midY = (y1 + y2) / 2;
    
    % Calculate the angle of the line segment (same as original function)
    a = (y2-y1) / (x2 - x1);
    rad = atan(a);
    degree = rad * (180/pi);
    
    % Rotation processing (same as original function)
    if degree > 0
        [RotM, newY, newX] = getRotatedM(im, (degree)-90);
    else
        [RotM, newY, newX] = getRotatedM(im, 90 + (degree));
    end
    
    RotatedImage = zeros(round(newY), round(newX));
    
    % Rotate the midpoint coordinates
    PointLineMid = [midY, midX, 1]';
    RotatedLineMid = round(rotatePM(PointLineMid, RotM));
    
    % Create a cropping box centered at the rotated midpoint
    halfWidth = boxWidth / 2;
    halfHeight = boxHeight / 2;
    
    % Create rectangle cropping coordinates [top-left y, top-left x, bottom-right y, bottom-right x]
    cropCoords = [
        RotatedLineMid(1,1)-halfHeight, RotatedLineMid(2,1)-halfWidth, ...
        RotatedLineMid(1,1)+halfHeight, RotatedLineMid(2,1)+halfWidth
    ];
    
    % Call the cropping function
    [imCroped] = ExtractPixelBox3_RGB(...
        cropCoords(1), cropCoords(2), boxWidth, boxHeight, ...
        RotatedImage, im, RotM);
end