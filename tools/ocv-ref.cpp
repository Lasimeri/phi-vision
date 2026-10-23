// ocv-ref: OpenCV's answer for one raw BGR frame, the oracle for lib/yunet.c.
// Writes OUTDIR/<head>.f32 for the 12 YuNet heads (the network run as FaceDetectorYN runs it:
// zero-padded to a multiple of 32, blobFromImage) and OUTDIR/faces.txt (FaceDetectorYN::detect rows).
// usage: ocv-ref MODEL W H FRAME.bgr OUTDIR [SCORE NMS TOPK]
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: ocv-ref MODEL W H FRAME.bgr OUTDIR [SCORE NMS TOPK]\n"); return 2; }
    const int W = atoi(argv[2]), H = atoi(argv[3]);
    const float score = argc > 6 ? (float)atof(argv[6]) : 0.6f, nms = argc > 7 ? (float)atof(argv[7]) : 0.3f;
    const int topk = argc > 8 ? atoi(argv[8]) : 50;
    cv::Mat img(H, W, CV_8UC3);
    FILE* f = std::fopen(argv[4], "rb");
    if (!f || std::fread(img.data, 1, img.total() * 3, f) != img.total() * 3) { std::fprintf(stderr, "cannot read %s\n", argv[4]); return 1; }
    std::fclose(f);
    const std::string out = argv[5];

    const int pw = ((W - 1) / 32 + 1) * 32, ph = ((H - 1) / 32 + 1) * 32;
    cv::Mat pad;
    cv::copyMakeBorder(img, pad, 0, ph - H, 0, pw - W, cv::BORDER_CONSTANT, 0);
    cv::dnn::Net net = cv::dnn::readNet(argv[1]);
    if (!net.getMainGraph()) net.setInputShape("input", cv::MatShape({1, 3, ph, pw}));   // old engine only, as face_detect.cpp
    net.setInput(cv::dnn::blobFromImage(pad));
    std::vector<cv::String> names = { "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
                                      "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32" };
    std::vector<cv::Mat> heads;
    net.forward(heads, names);
    for (size_t i = 0; i < names.size(); i++) {
        cv::Mat h = heads[i].isContinuous() ? heads[i] : heads[i].clone();
        FILE* o = std::fopen((out + "/" + names[i] + ".f32").c_str(), "wb");
        std::fwrite(h.data, sizeof(float), h.total(), o);
        std::fclose(o);
    }

    auto yn = cv::FaceDetectorYN::create(argv[1], "", cv::Size(W, H), score, nms, topk);
    cv::Mat det;
    yn->detect(img, det);
    FILE* o = std::fopen((out + "/faces.txt").c_str(), "w");
    for (int i = 0; i < det.rows; i++)
        for (int k = 0; k < 15; k++) std::fprintf(o, "%.9g%c", det.at<float>(i, k), k == 14 ? '\n' : ' ');
    std::fclose(o);
    std::printf("%d faces\n", det.rows);
    return 0;
}
