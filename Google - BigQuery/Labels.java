/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/Labels.java
 *
 *  Created on: Feb 14, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import com.google.api.client.util.Data;
import com.google.auto.value.AutoValue;
import com.google.auto.value.extension.memoized.Memoized;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

@AutoValue
abstract class Labels implements Serializable 
{
    private static final long serialVersionUID = 1L;
    static final Labels ZERO = of(Collections.<String, String>emptyMap());
    
    @Nullable
    abstract Map<String, String> userMap();

    @Memoized
    @Nullable
    Map<String, String> toPb() 
    {
        Map<String, String> userMap = userMap();
        if (userMap == null)
        {
            return Data.nullOf(HashMap.class);
        }
        if (userMap.isEmpty())
        {
            return null;
        }
        HashMap<String, String> pbMap = new HashMap<>();
        for (Map.Entry<String, String> entry : userMap.entrySet())
        {
            String key = entry.getKey();
            String val = entry.getValue();
            if (val == null)
            {
                val = Data.NULL_STRING;
            }
            pbMap.put(key, val);
        }
        return Collections.unmodifiableMap(pbMap);
    }

    private static Labels of(Map<String, String> userMap)
    {
        Preconditions.checkArgument(
            userMap == null || !userMap.containsKey(null), "null keys are not supported"
        );
        return AutoValue_Labels(userMap);
    }

    static Labels fromUser(Map<String, String> map)
    {
        if (map == null || map instanceof ImmutableMap)
        {
            return of(map);
        }
        return of(Collections.unmodifiableMap(new HashMap<>(map)));
    }

    static Labels fromPb(Map<String, String> pb)
    {
        if (Data.isNull(pb))
        {
            return of(null);
        }
        if (pb == null || pb.isEmpty())
        {
            return of(Collections.<String, String>emptyMap());
        }

        HashMap<String, String> map = new HashMap<>();
        for (Map.Entry<String, String> entry : pb.entrySet())
        {
            String key = entry.getKey();
            String val = entry.getValue();
            if (Data.isNull(val))
            {
                val = null;
            }
            map.put(key, val);
        }
        return of(Collections.unmodifiableMap(map));
    }
}
