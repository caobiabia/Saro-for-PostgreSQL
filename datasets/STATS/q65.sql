select  count(*) from comments as c,  		posts as p,          postHistory as ph,          votes as v,          badges as b,          users as u  where u.Id = p.OwnerUserId     and u.Id = b.UserId     and p.Id = c.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND b.Date<='2014-09-03 22:43:25'::timestamp  AND c.CreationDate>='2010-10-05 00:53:11'::timestamp  AND c.CreationDate<='2014-09-11 14:13:51'::timestamp  AND ph.PostHistoryTypeId=3  AND p.Score<=17  AND p.ViewCount>=0  AND p.ViewCount<=12291  AND u.Reputation<=1001  AND u.Views>=0;