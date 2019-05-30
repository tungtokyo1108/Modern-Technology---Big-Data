/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-data-lake-store-java/blob/master/src/main/java/com/microsoft/azure/datalake/store/acl/AclEntry.java
 *
 *  Created on: May 28, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.datalake.store.mirror.acl;

import java.util.LinkedList;
import java.util.List;

public class AclEntry 
{
    public AclScope scope;
    public AclType type;

    public String name;
    public AclAction action;

    public AclEntry() {}

    public AclEntry(AclScope scope, AclType type, String name, AclAction action)
    {
        if (scope == null) throw new IllegalArgumentException("AclScope is null");
        if (type == null) throw new IllegalArgumentException("AclType is null");
        if (type == AclType.MASK && name != null && !name.trim().equals(""))
        {
            throw new IllegalArgumentException("mask should not have user/group name");
        }
        if (type == AclType.OTHER && name != null && !name.trim().equals(""))
        {
            throw new IllegalArgumentException("ACL entry type 'other' should not have user/group name");
        }

        this.scope = scope;
        this.type = type;
        this.name = name;
        this.action = action;
    }

    public static AclEntry parseAclEntry(String entryString) throws IllegalArgumentException {
        return parseAclEntry(entryString, false);
    }

    public static AclEntry parseAclEntry(String entryString, boolean removeAcl) throws IllegalArgumentException {
        if (entryString == null || entryString.equals("")) return null;
        AclEntry aclEntry = new AclEntry();
        String aclString = entryString.trim();

        // remaining string should have 3 entries 
        String[] parts = aclString.split(":", 5);
        if (parts.length < 2 || parts.length > 3) throw new IllegalArgumentException("Invalid AclEntryString " + entryString);
        if (parts.length == 2 && !removeAcl) throw new IllegalArgumentException("Invalid AclEntryString " + entryString);

        // entry type 
        try {
            aclEntry.type = AclType.valueOf(parts[0].toUpperCase().trim());
        } catch (IllegalArgumentException ex) {
            throw new IllegalArgumentException("Invalid Acl AclType in " + entryString);
        } catch (NullPointerException ex) {
            throw new IllegalArgumentException("ACL Entry AclType missing in " + entryString);
        }

        // user/group name 
        aclEntry.name = parts[1].trim();
        if (aclEntry.type == AclType.MASK && !aclEntry.name.equals(""))
        {
            throw new IllegalArgumentException("Mask entry cannot user/group name: " + entryString);
        }
        if (aclEntry.type == AclType.MASK && !aclEntry.name.equals(""))
        {
            throw new IllegalArgumentException("Entry of type 'other' should not contain user/group name: " + entryString);
        }

        // permission
        if (!removeAcl)
        {
            try {
                aclEntry.action = AclAction.fromRwx(parts[2]);
            } catch (IllegalArgumentException ex) {
                throw new IllegalArgumentException("Invalid ACL action in " + entryString);
            } catch (NullPointerException ex) {
                throw new IllegalArgumentException("ACL action missing in " + entryString);
            }
        }

        return aclEntry;
    }

    public String toString()
    {
        return this.toString(false);
    }

    public String toString(boolean removeAcl)
    {
        StringBuilder str = new StringBuilder();
        if (this.scope == null) throw new IllegalArgumentException("Acl Entry has no scope");
        if (this.type == null) throw new IllegalArgumentException("Acl Entry has no type");

        str.append(this.type.toString().toLowerCase());
        str.append(":");

        if (this.name != null) str.append(this.name);

        if (this.action != null && !removeAcl)
        {
            str.append(":");
            str.append(this.action.toString());
        }
        return str.toString();
    }

    public static List<AclEntry> parseAclSpec(String aclString) throws IllegalArgumentException 
    {
        if (aclString == null || aclString.trim().equals("")) return new LinkedList<AclEntry>();

        aclString = aclString.trim();
        String car; // The first entry 
        String cdr; // The rest of the list after first entry 

        int commaPos = aclString.indexOf(",");
        if (commaPos < 0)
        {
            car = aclString;
            cdr = null;
        }
        else 
        {
            car = aclString.substring(0, commaPos).trim();
            cdr = aclString.substring(commaPos+1);
        }
        LinkedList<AclEntry> aclSpec = (LinkedList<AclEntry>) parseAclSpec(cdr);
        if (!car.equals(""))
        {
            aclSpec.addFirst(parseAclEntry(car));
        }
        return aclSpec;
    }

    public static String aclListToString(List<AclEntry> list)
    {
        return aclListToString(list, false);
    }

    public static String aclListToString(List<AclEntry> list, boolean removeAcl)
    {
        if (list == null || list.size() == 0) return "";

        String separator = "";
        StringBuilder output = new StringBuilder();

        for (AclEntry entry : list)
        {
            output.append(separator);
            output.append(entry.toString(removeAcl));
            separator = ",";
        }
        return output.toString();
    }
}
