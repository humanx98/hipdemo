
set(HEADERS
    include/ww/viewport.h
)

set(SOURCES
    viewport.c
)

add_library(viewport STATIC ${HEADERS} ${SOURCES})
target_include_directories(viewport PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(viewport ww_core)