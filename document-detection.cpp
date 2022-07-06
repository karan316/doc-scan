Vec3f calcParams(Point2f p1, Point2f p2) // line's equation Params computation
{
    float a, b, c;
    if (p2.y - p1.y == 0)
    {
        a = 0.0f;
        b = -1.0f;
    }
    else if (p2.x - p1.x == 0)
    {
        a = -1.0f;
        b = 0.0f;
    }
    else
    {
        a = (p2.y - p1.y) / (p2.x - p1.x);
        b = -1.0f;
    }

    c = (-a * p1.x) - b * p1.y;
    return(Vec3f(a, b, c));
}

Point findIntersection(Vec3f params1, Vec3f params2)
{
    float x = -1, y = -1;
    float det = params1[0] * params2[1] - params2[0] * params1[1];
    if (det < 0.5f && det > -0.5f) // lines are approximately parallel
    {
        return(Point(-1, -1));
    }
    else
    {
        x = (params2[1] * -params1[2] - params1[1] * -params2[2]) / det;
        y = (params1[0] * -params2[2] - params2[0] * -params1[2]) / det;
    }
    return(Point(x, y));
}

vector<Point> getQuadrilateral(Mat & grayscale, Mat& output) // returns that 4 intersection points of the card
{
    Mat convexHull_mask(grayscale.rows, grayscale.cols, CV_8UC1);
    convexHull_mask = Scalar(0);

    vector<vector<Point>> contours;
    findContours(grayscale, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    vector<int> indices(contours.size());
    iota(indices.begin(), indices.end(), 0);

    sort(indices.begin(), indices.end(), [&contours](int lhs, int rhs) {
        return contours[lhs].size() > contours[rhs].size();
    });

    /// Find the convex hull object
    vector<vector<Point> >hull(1);
    convexHull(Mat(contours[indices[0]]), hull[0], false);

    vector<Vec4i> lines;
    drawContours(convexHull_mask, hull, 0, Scalar(255));
    imshow("convexHull_mask", convexHull_mask);
    HoughLinesP(convexHull_mask, lines, 1, CV_PI / 200, 50, 50, 10);
    cout << "lines size:" << lines.size() << endl;

    if (lines.size() == 4) // we found the 4 sides
    {
        vector<Vec3f> params(4);
        for (int l = 0; l < 4; l++)
        {
            params.push_back(calcParams(Point(lines[l][0], lines[l][1]), Point(lines[l][2], lines[l][3])));
        }

        vector<Point> corners;
        for (int i = 0; i < params.size(); i++)
        {
            for (int j = i; j < params.size(); j++) // j starts at i so we don't have duplicated points
            {
                Point intersec = findIntersection(params[i], params[j]);
                if ((intersec.x > 0) && (intersec.y > 0) && (intersec.x < grayscale.cols) && (intersec.y < grayscale.rows))
                {
                    cout << "corner: " << intersec << endl;
                    corners.push_back(intersec);
                }
            }
        }

        for (int i = 0; i < corners.size(); i++)
        {
            circle(output, corners[i], 3, Scalar(0, 0, 255));
        }

        if (corners.size() == 4) // we have the 4 final corners
        {
            return(corners);
        }
    }
    
    return(vector<Point>());
}

int main(int argc, char** argv)
{
    Mat input = imread("playingcard_input.png");
    Mat input_grey;
    cvtColor(input, input_grey, CV_BGR2GRAY);
    Mat threshold1;
    Mat edges;
    blur(input_grey, input_grey, Size(3, 3));
    Canny(input_grey, edges, 30, 100);

    vector<Point> card_corners = getQuadrilateral(edges, input);
    Mat warpedCard(400, 300, CV_8UC3);
    if (card_corners.size() == 4)
    {
        Mat homography = findHomography(card_corners, vector<Point>{Point(warpedCard.cols, 0), Point(warpedCard.cols, warpedCard.rows), Point(0,0) , Point(0, warpedCard.rows)});
        warpPerspective(input, warpedCard, homography, Size(warpedCard.cols, warpedCard.rows));
    }

    imshow("warped card", warpedCard);
    imshow("edges", edges);
    imshow("input", input);
    waitKey(0);

    return 0;
}