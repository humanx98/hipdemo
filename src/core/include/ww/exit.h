#pragma once

#include <stdlib.h>
#include <ww/log.h>

#define WW_EXIT                                             \
    do {                                                    \
        WW_LOG_ERROR("%s:%d: exit.\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                 \
    } while (0)

#define WW_EXIT_WITH_MSG(format, ...)                                      \
    do {                                                                   \
        WW_LOG_ERROR("%s:%d: exit with message.\n", __FILE__, __LINE__);   \
        WW_LOG_ERROR(format, __VA_ARGS__);                                 \
        exit(EXIT_FAILURE);                                                \
    } while (0)
