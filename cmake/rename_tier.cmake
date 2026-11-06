# cmake -DNM=<nm> -DOBJCOPY=<objcopy> -DTAG=<v3|v4> -DLIST=<file> -DOUT=<dir> -P rename_tier.cmake
# LIST holds one tier object per line. Each symbol that a tier object defines, except
# tier_entry and @feat.00, gets the prefix admiral_<TAG>_. One map covers all objects of the
# tier, so a tier reference follows its definition. The census is every defined symbol, not
# only the external ones: a local COMDAT group signature needs the prefix too, or the link
# dedups one tier's groups against another's and drops the tier's definitions.
cmake_minimum_required(VERSION 3.25)
file(STRINGS "${LIST}" objects)
execute_process(COMMAND "${NM}" --defined-only --format=just-symbols ${objects}
                OUTPUT_VARIABLE out COMMAND_ERROR_IS_FATAL ANY)
if(out MATCHES "[][;]")
    message(FATAL_ERROR "rename_tier: a name holds [, ] or ;, which a CMake list cannot carry")
endif()
string(REPLACE "\n" ";" names "${out}")
list(FILTER names EXCLUDE REGEX ":$|^$|tier_entry|@feat\\.00")
list(REMOVE_DUPLICATES names)
list(SORT names)
list(LENGTH names n)
if(n EQUAL 0)
    message(FATAL_ERROR "rename_tier: tier ${TAG} defines no external name")
endif()
list(TRANSFORM names REPLACE "^.+$" "\\0 admiral_${TAG}_\\0")
list(JOIN names "\n" map)
file(MAKE_DIRECTORY "${OUT}")
file(WRITE "${OUT}/map" "${map}\n")
foreach(o IN LISTS objects)
    get_filename_component(b "${o}" NAME)
    execute_process(COMMAND "${OBJCOPY}" "--redefine-syms=${OUT}/map" "${o}" "${OUT}/${b}"
                    COMMAND_ERROR_IS_FATAL ANY)
endforeach()
