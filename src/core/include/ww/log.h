#pragma once

#include <stdio.h>

#define WW_LOG_GRN_COLOR   "\x1B[32m"
#define WW_LOG_WHT_COLOR   "\x1B[37m"
#define WW_LOG_YEL_COLOR   "\x1B[33m"
#define WW_LOG_RED_COLOR   "\x1B[31m"
#define WW_LOG_RESET_COLOR "\x1B[0m"

#define WW_LOG_DEBUG(format, ...) fprintf(stdout, WW_LOG_GRN_COLOR "[DEBUG] " format WW_LOG_RESET_COLOR __VA_OPT__(,) __VA_ARGS__)
#define WW_LOG_INFO(format, ...) fprintf(stdout, WW_LOG_WHT_COLOR "[INFO] " format WW_LOG_RESET_COLOR __VA_OPT__(,) __VA_ARGS__)
#define WW_LOG_WARN(format, ...) fprintf(stderr, WW_LOG_YEL_COLOR "[WARN] " format WW_LOG_RESET_COLOR __VA_OPT__(,) __VA_ARGS__)
#define WW_LOG_ERROR(format, ...) fprintf(stderr, WW_LOG_RED_COLOR "[ERROR] " format WW_LOG_RESET_COLOR __VA_OPT__(,) __VA_ARGS__)
