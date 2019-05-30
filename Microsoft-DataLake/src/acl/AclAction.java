/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/acl/AclAction.java
 *
 *  Created on: May 29, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror.acl;

import java.security.acl.Acl;

public enum AclAction 
{
    NONE ("---"),
    EXECUTE ("--x"),
    WRITE ("-w-"),
    WRITE_EXECUTE ("-wx"),
    READ ("r--"),
    READ_EXECUTE ("r-x"),
    READ_WRITE ("rw-"),
    ALL ("rwx");

    private final String rwx;
    private static final AclAction[] values = AclAction.values();

    AclAction(String rwx)
    {
        this.rwx = rwx;
    }

    public static String toString(AclAction action)
    {
        return action.rwx;
    }

    public static AclAction fromRwx(String rwx)
    {
        if (rwx == null) throw new IllegalArgumentException("access specifier is null");
        rwx = rwx.trim().toLowerCase();
        for (AclAction a: values)
        {
            if (a.rwx.equals(rwx)) {
                return a;
            }
        }
        throw new IllegalArgumentException(rwx + "is not a valid access specifier");
    }

    public static boolean isValidRwx(String input)
    {
        try {
            fromRwx(input);
            return true;
        } catch (IllegalArgumentException ex) {
            return false;
        }
    }

    public static AclAction fromOctal(int perm) {
        if (perm < 0 || perm > 7) throw new IllegalArgumentException(perm + " is not a valid access specifier");
        return values[perm];
    }

    public int toOctal() {
        return this.ordinal();
    }

    public static int toOctal(AclAction action) 
    {
        return action.ordinal();
    }
}
