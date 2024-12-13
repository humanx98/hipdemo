#include <ww/file.h>
#include <ww/log.h>
#include <stdio.h>
#include <stdlib.h>

b8 ww_read_file_to_darray(const char* file, WwDArray(u8)* darr) {
    assert(file);
    assert(darr);

    FILE *fp = fopen(file, "rb");
    if (fp == NULL) {
        WW_LOG_ERROR("Couldn't open file: %s\n", file);
        return false;
    }

    if (fseek(fp, 0L, SEEK_END) != 0) {
        goto failed;
    }

    i64 bufsize = ftell(fp);
    if (bufsize == -1) {
        goto failed;
    }

    if (!ww_darray_ensure_total_capacity_precise(darr, bufsize)) {
        WW_LOG_ERROR("Couldn't allocate enough memory to read file %s\n", file);
        goto failed;
    }
    ww_darray_resize_assume_capacity(darr, bufsize);

    if (fseek(fp, 0L, SEEK_SET) != 0) {
        goto failed;
    }

    if (ww_darray_len(darr) != fread(ww_darray_ptr(darr), ww_darray_elem_size(darr), bufsize, fp)) {
        WW_LOG_ERROR("Couldn't read the whole file: %s\n", file);
        goto failed;
    }

    if (ferror(fp) != 0) {
        WW_LOG_ERROR("Error reading file: %s\n", file);
    }

    fclose(fp);
    return true;
failed:
    fclose(fp);
    return false;
}
