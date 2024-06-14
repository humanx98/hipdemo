#include "ww/collections/darray.h"
#include <ww/file.h>
#include <ww/log.h>
#include <stdio.h>
#include <stdlib.h>

b8 ww_read_file_to_darray(const char* file, WwDArray(u8)* darr) {
    assert(file);
    assert(darr);

    FILE *fp = fopen(file, "r");
    b8 err = fp == NULL;
    if (err) {
        WW_LOG_ERROR("Couldn't open file: %s\n", file);
        goto failed;
    }

    err = fseek(fp, 0L, SEEK_END) != 0;
    if (err) {
        goto failed;
    }

    i64 bufsize = ftell(fp);
    err = bufsize == -1;
    if (err) {
        goto failed;
    }

    err = !ww_darray_ensure_total_capacity_precise(darr, bufsize);
    if (err) {
        WW_LOG_ERROR("Couldn't allocate enough memory to read file %s\n", file);
        goto failed;
    }
    ww_darray_resize_assume_capacity(darr, bufsize);


    err = fseek(fp, 0L, SEEK_SET) != 0;
    if (err) {
        goto failed;
    }

    err = ww_darray_len(darr) != fread(ww_darray_ptr(darr), ww_darray_elem_size(darr), bufsize, fp);
    if (err) {
        WW_LOG_ERROR("Couldn't read the whole file: %s\n", file);
        goto failed;
    }

    err = ferror(fp) != 0;
    if (err) {
        WW_LOG_ERROR("Error reading file: %s\n", file);
    }
failed:
    if (fp != NULL) {
        fclose(fp);
    }
    return !err;
}
