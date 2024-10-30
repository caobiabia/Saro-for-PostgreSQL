select  count(*) from comments as c,  		votes as v where c.UserId = v.UserId  AND c.Score=0  AND v.VoteTypeId=3;
